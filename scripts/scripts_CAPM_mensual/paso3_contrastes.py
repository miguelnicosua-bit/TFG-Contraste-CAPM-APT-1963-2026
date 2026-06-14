"""
PASO 3 - Contrastes formales del CAPM
TFG Valoración de Activos - Contraste empírico del CAPM
Autor: Miguel Suárez Crespo

Cuatro bloques de contrastes:

  1. TEST GRS (Gibbons-Ross-Shanken, 1989):
     H0: alpha_j = 0 para todas las N=100 carteras conjuntamente. Aplicado
     con las estimaciones de la primera etapa ESTÁTICA, que es la
     formulación canónica del test (CLM 1997, cap. 5.4).

  2. TESTS DE HETEROCEDASTICIDAD (Breusch-Pagan y White) sobre los residuos
     de las 100 regresiones de la primera etapa ESTÁTICA. Estos contrastes
     justifican el uso de la versión móvil (paso 1B): si los residuos
     presentan heterocedasticidad, las betas estáticas no son eficientes y
     la metodología con ventana móvil tiene sentido como corrección.

  3. CONTRASTE DE LINEALIDAD: añadir beta^2 a la regresión transversal con
     betas móviles. Bajo CAPM, gamma_2 = 0.

  4. CONTRASTE DE REMUNERACIÓN DEL RIESGO IDIOSINCRÁTICO: añadir
     sigma(epsilon) a la regresión transversal con betas móviles. Bajo
     CAPM, gamma_3 = 0.

Los contrastes 3 y 4 emplean las betas (y sigmas residuales) estimadas con
ventana móvil de 60 meses para coherencia con la metodología de la segunda
etapa.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smdiag
from scipy import stats

# ========================================================================
# CARGA DE DATOS
# ========================================================================
CARPETA_PROCESADOS = "datos_procesados"
CARPETA_RESULTADOS = "resultados/tablas/capm_mensual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
mkt_rf  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))
primera = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))
primera_rolling = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_rolling.pkl"))

betas_rolling        = primera_rolling["betas_rolling"]
sigma_resid_rolling  = primera_rolling["sigma_resid_rolling"]
VENTANA              = primera_rolling["ventana"]

print(f"Datos cargados: T = {excesos.shape[0]} meses, N = {excesos.shape[1]} carteras")
print(f"  - Ventana móvil: {VENTANA} meses")
print(f"  - Secciones cruzadas disponibles: {betas_rolling.shape[0]}")

# ========================================================================
# CONTRASTE 1: TEST GRS (Gibbons-Ross-Shanken, 1989)
# ========================================================================
# Aplicado con alfas y residuos ESTÁTICOS (formulación canónica).
print("\n" + "=" * 70)
print("CONTRASTE 1: TEST GRS (Gibbons-Ross-Shanken, 1989)")
print("=" * 70)
print("H0: alpha_j = 0 para todas las 100 carteras conjuntamente")

# Recalculamos los residuos de la primera etapa estática
residuos = pd.DataFrame(index=excesos.index, columns=excesos.columns, dtype=float)
X_t = sm.add_constant(mkt_rf)
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    residuos.loc[datos.index, cartera] = modelo.resid

residuos_grs = residuos.dropna()
T_grs = residuos_grs.shape[0]
N_grs = residuos_grs.shape[1]

alpha = primera["alpha"].values.reshape(-1, 1)
Sigma = residuos_grs.cov().values

mkt_grs = mkt_rf.loc[residuos_grs.index]
SR_M_squared = (mkt_grs.mean() / mkt_grs.std()) ** 2

Sigma_inv = np.linalg.inv(Sigma)
alpha_quad = (alpha.T @ Sigma_inv @ alpha).item()

grs_stat = ((T_grs - N_grs - 1) / N_grs) * alpha_quad / (1 + SR_M_squared)
grs_pvalue = 1 - stats.f.cdf(grs_stat, dfn=N_grs, dfd=T_grs - N_grs - 1)

print(f"\nT = {T_grs} meses, N = {N_grs} carteras")
print(f"Estadístico GRS:        {grs_stat:.4f}")
print(f"Distribución bajo H0:   F({N_grs}, {T_grs - N_grs - 1})")
print(f"p-valor:                {grs_pvalue:.6f}")
print(f"Sharpe^2 mercado:       {SR_M_squared:.4f}")
print(f"alpha' Sigma^-1 alpha:  {alpha_quad:.4f}")
if grs_pvalue < 0.05:
    print(f"\n→ Se RECHAZA H0 al 5%. Los alfas son CONJUNTAMENTE distintos de cero.")
else:
    print(f"\n→ No se rechaza H0. Alfas conjuntamente compatibles con cero.")

# ========================================================================
# CONTRASTE 2: TESTS DE HETEROCEDASTICIDAD SOBRE LA 1ª ETAPA ESTÁTICA
# ========================================================================
# Para cada una de las 100 regresiones de la primera etapa estática
# (modelo de mercado sobre los 754 meses), se aplican dos tests:
#
#   - Breusch-Pagan: H0 los residuos son homocedásticos.
#     Estadístico LM ~ chi^2(k), donde k = número de regresores
#     (sin contar la constante). Aquí k=1.
#   - White: versión más general. Estadístico ~ chi^2 con grados de libertad
#     que dependen de la especificación auxiliar.
#
# Razón económica (Sampayo): la versión móvil corrige la heterocedasticidad
# que se observe en los residuos de la versión estática. Si la mayoría de
# las 100 carteras rechazan homocedasticidad, queda justificada la
# metodología de ventana móvil.
print("\n" + "=" * 70)
print("CONTRASTE 2: HETEROCEDASTICIDAD DE LA 1ª ETAPA ESTÁTICA")
print("=" * 70)
print("Tests aplicados a los residuos de cada una de las 100 regresiones")
print("del modelo de mercado (paso 1A) sobre los 754 meses del muestral.")
print("\nH0 (en ambos tests): los residuos son homocedásticos.")

X_t_bp = sm.add_constant(mkt_rf)
resultados_hetero = []

for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t_bp], axis=1).dropna()
    Xc = datos[["const", "Mkt-RF"]]
    yc = datos[cartera]
    modelo = sm.OLS(yc, Xc).fit()

    # Breusch-Pagan
    bp_lm, bp_lm_p, bp_f, bp_f_p = smdiag.het_breuschpagan(modelo.resid, Xc)
    # White
    w_lm, w_lm_p, w_f, w_f_p = smdiag.het_white(modelo.resid, Xc)

    resultados_hetero.append({
        "cartera":      cartera,
        "BP_LM":        bp_lm,
        "BP_p":         bp_lm_p,
        "BP_rechaza":   int(bp_lm_p < 0.05),
        "White_LM":     w_lm,
        "White_p":      w_lm_p,
        "White_rechaza": int(w_lm_p < 0.05),
    })

hetero_df = pd.DataFrame(resultados_hetero).set_index("cartera")

n_bp_5    = hetero_df["BP_rechaza"].sum()
n_bp_1    = (hetero_df["BP_p"] < 0.01).sum()
n_white_5 = hetero_df["White_rechaza"].sum()
n_white_1 = (hetero_df["White_p"] < 0.01).sum()

print(f"\nResultados sobre 100 carteras:")
print(f"  {'Test':<15} {'Rechazan al 5%':>16} {'Rechazan al 1%':>16}")
print(f"  {'-'*47}")
print(f"  {'Breusch-Pagan':<15} {n_bp_5:>16} / 100  {n_bp_1:>16} / 100")
print(f"  {'White':<15} {n_white_5:>16} / 100  {n_white_1:>16} / 100")

print(f"\nValores medios de los p-valores:")
print(f"  Breusch-Pagan: {hetero_df['BP_p'].mean():.4f}")
print(f"  White:         {hetero_df['White_p'].mean():.4f}")

print(f"\nValores medianos:")
print(f"  Breusch-Pagan: {hetero_df['BP_p'].median():.4f}")
print(f"  White:         {hetero_df['White_p'].median():.4f}")

if n_bp_5 > 50 or n_white_5 > 50:
    print(f"\n→ Más de la mitad de las carteras presentan heterocedasticidad")
    print(f"  significativa al 5%. Esto justifica el uso de la metodología")
    print(f"  con ventana móvil de 60 meses, que permite que las betas")
    print(f"  varíen en el tiempo y atenúa el problema.")
else:
    print(f"\n→ Heterocedasticidad detectada en una proporción minoritaria de")
    print(f"  las carteras. La metodología con ventana móvil sigue siendo")
    print(f"  preferible por razones de robustez.")

# ========================================================================
# CONTRASTE 3: LINEALIDAD (gamma_2 sobre beta^2)
# ========================================================================
# Para cada mes t, regresión transversal:
#   r_{j,t} - r_{f,t} = gamma_0,t + gamma_1,t * beta_{j,t-1}
#                       + gamma_2,t * beta_{j,t-1}^2 + u_{j,t}
print("\n" + "=" * 70)
print("CONTRASTE 3: LINEALIDAD (Fama-MacBeth con beta y beta^2)")
print("=" * 70)
print("H0: gamma_2 = 0 en la regresión:")
print("    r_{i,t} - r_f,t = gamma_0,t + gamma_1,t beta_i + gamma_2,t beta_i^2 + u")
print("(betas con ventana móvil de 60 meses)")

gammas_lin_list = []

for fecha_t in betas_rolling.index:
    y      = excesos.loc[fecha_t]
    beta_t = betas_rolling.loc[fecha_t]
    beta_sq_t = beta_t ** 2

    X = pd.concat([
        pd.Series(1.0, index=beta_t.index, name="const"),
        beta_t.rename("beta"),
        beta_sq_t.rename("beta_sq"),
    ], axis=1)

    datos = pd.concat([y, X], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_lin_list.append(modelo.params.values)

gammas_lin = pd.DataFrame(gammas_lin_list, columns=["gamma_0", "gamma_1", "gamma_2"])

T_lin  = len(gammas_lin)
medias  = gammas_lin.mean()
se      = gammas_lin.std() / np.sqrt(T_lin)
t_stats = medias / se
p_vals  = pd.Series(2 * (1 - stats.t.cdf(t_stats.abs(), df=T_lin-1)), index=t_stats.index)

tabla_lin = pd.DataFrame({
    "media":         medias,
    "media_anual_%": medias * 12 * 100,
    "error_std":     se,
    "t_stat":        t_stats,
    "p_valor":       p_vals,
})
print("\n" + tabla_lin.round(5).to_string())

g2_p = p_vals["gamma_2"]
print(f"\nContraste H0: gamma_2 = 0  (p-valor = {g2_p:.4f})")
if g2_p < 0.05:
    print(f"→ Se RECHAZA H0 al 5%. La relación rendimiento-beta NO es lineal.")
else:
    print(f"→ No se rechaza H0. Compatible con linealidad.")

# ========================================================================
# CONTRASTE 4: RIESGO IDIOSINCRÁTICO (gamma_3 sobre sigma(epsilon))
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE 4: RIESGO IDIOSINCRÁTICO")
print("=" * 70)
print("H0: gamma_3 = 0 en la regresión:")
print("    r_{i,t} - r_f,t = gamma_0,t + gamma_1,t beta_i + gamma_3,t sigma(e_i) + u")
print("(betas y sigmas con ventana móvil de 60 meses)")

gammas_idio_list = []

for fecha_t in betas_rolling.index:
    y       = excesos.loc[fecha_t]
    beta_t  = betas_rolling.loc[fecha_t]
    sigma_t = sigma_resid_rolling.loc[fecha_t]

    X = pd.concat([
        pd.Series(1.0, index=beta_t.index, name="const"),
        beta_t.rename("beta"),
        sigma_t.rename("sigma_resid"),
    ], axis=1)

    datos = pd.concat([y, X], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_idio_list.append(modelo.params.values)

gammas_idio = pd.DataFrame(gammas_idio_list, columns=["gamma_0", "gamma_1", "gamma_3"])

T_idio  = len(gammas_idio)
medias  = gammas_idio.mean()
se      = gammas_idio.std() / np.sqrt(T_idio)
t_stats = medias / se
p_vals  = pd.Series(2 * (1 - stats.t.cdf(t_stats.abs(), df=T_idio-1)), index=t_stats.index)

tabla_idio = pd.DataFrame({
    "media":         medias,
    "media_anual_%": medias * 12 * 100,
    "error_std":     se,
    "t_stat":        t_stats,
    "p_valor":       p_vals,
})
print("\n" + tabla_idio.round(5).to_string())

g3_p = p_vals["gamma_3"]
print(f"\nContraste H0: gamma_3 = 0  (p-valor = {g3_p:.4f})")
if g3_p < 0.05:
    print(f"→ Se RECHAZA H0 al 5%. El riesgo idiosincrático SÍ está valorado.")
else:
    print(f"→ No se rechaza H0. El riesgo idiosincrático no parece tener precio.")

# ========================================================================
# RESUMEN GLOBAL
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN DE LOS CUATRO BLOQUES DE CONTRASTES")
print("=" * 70)
print(f"{'Contraste':<40} {'Estadístico':>12} {'p-valor':>10} {'Decisión':>15}")
print("-" * 80)
print(f"{'GRS (alfas conjuntos = 0)':<40} {grs_stat:>12.3f} {grs_pvalue:>10.4f} "
      f"{'Rechaza CAPM' if grs_pvalue < 0.05 else 'No rechaza':>15}")
print(f"{'Heterocedasticidad BP (% rechazan al 5%)':<40} "
      f"{n_bp_5:>10}/100  {'':>10} "
      f"{'Justifica móvil' if n_bp_5 > 50 else 'Sin evidencia':>15}")
print(f"{'Heterocedasticidad W. (% rechazan al 5%)':<40} "
      f"{n_white_5:>10}/100  {'':>10} "
      f"{'Justifica móvil' if n_white_5 > 50 else 'Sin evidencia':>15}")
print(f"{'Linealidad (gamma_2 = 0)':<40} {tabla_lin.loc['gamma_2','t_stat']:>12.3f} "
      f"{tabla_lin.loc['gamma_2','p_valor']:>10.4f} "
      f"{'Rechaza CAPM' if tabla_lin.loc['gamma_2','p_valor'] < 0.05 else 'No rechaza':>15}")
print(f"{'Riesgo idiosincrático (gamma_3 = 0)':<40} {tabla_idio.loc['gamma_3','t_stat']:>12.3f} "
      f"{tabla_idio.loc['gamma_3','p_valor']:>10.4f} "
      f"{'Rechaza CAPM' if tabla_idio.loc['gamma_3','p_valor'] < 0.05 else 'No rechaza':>15}")

# ========================================================================
# GUARDADO
# ========================================================================
tabla_lin.to_csv(os.path.join(CARPETA_RESULTADOS, "contraste_linealidad.csv"))
tabla_idio.to_csv(os.path.join(CARPETA_RESULTADOS, "contraste_idiosincratico.csv"))
hetero_df.to_csv(os.path.join(CARPETA_RESULTADOS, "test_heterocedasticidad.csv"))

resumen_grs = pd.DataFrame({
    "estadistico":  [grs_stat],
    "p_valor":      [grs_pvalue],
    "T":            [T_grs],
    "N":            [N_grs],
    "Sharpe2_mkt":  [SR_M_squared],
    "alpha_quad":   [alpha_quad],
})
resumen_grs.to_csv(os.path.join(CARPETA_RESULTADOS, "test_grs.csv"), index=False)

resumen_hetero = pd.DataFrame({
    "test":              ["Breusch-Pagan", "White"],
    "rechazan_al_5%":    [n_bp_5, n_white_5],
    "rechazan_al_1%":    [n_bp_1, n_white_1],
    "p_medio":           [hetero_df['BP_p'].mean(), hetero_df['White_p'].mean()],
    "p_mediano":         [hetero_df['BP_p'].median(), hetero_df['White_p'].median()],
})
resumen_hetero.to_csv(os.path.join(CARPETA_RESULTADOS, "resumen_heterocedasticidad.csv"),
                     index=False)

print(f"\n✓ Tablas exportadas a '{CARPETA_RESULTADOS}/'")
print("✓ Listos para ejecutar paso4_graficos.py")
