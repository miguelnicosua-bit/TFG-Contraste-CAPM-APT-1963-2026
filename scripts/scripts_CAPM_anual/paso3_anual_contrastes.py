"""
PASO 3 ANUAL - Contrastes formales del CAPM (frecuencia anual)
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Replica el paso 3 mensual con cuatro bloques de contrastes:

  1. TEST GRS (Gibbons-Ross-Shanken, 1989):
     LIMITACIÓN: con datos anuales T=62 y N=100, el test GRS no es
     aplicable directamente porque requiere T >= N para invertir la matriz
     de covarianzas residuales Sigma (N x N). Se documenta la limitación
     y se aplica una versión alternativa con T-N efectivo reducido si es
     viable; en otro caso se omite.

  2. TESTS DE HETEROCEDASTICIDAD (White) sobre los residuos de las 100
     regresiones del modelo de mercado anual (paso 1A).

  3. CONTRASTE DE LINEALIDAD: añadir beta^2 a la regresión transversal
     con betas rolling. Bajo CAPM, gamma_2 = 0.

  4. CONTRASTE DE RIESGO IDIOSINCRÁTICO: añadir sigma(epsilon) a la
     regresión transversal con betas rolling. Bajo CAPM, gamma_3 = 0.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smdiag
from scipy import stats

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")
CARPETA_RESULTADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "resultados", "tablas", "capm_anual")
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_anual.pkl"))
mkt_rf  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf_anual.pkl"))
primera = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_anual.pkl"))
primera_rolling = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_anual_rolling.pkl"))

betas_rolling       = primera_rolling["betas_rolling"]
sigma_resid_rolling = primera_rolling["sigma_resid_rolling"]
VENTANA             = primera_rolling["ventana"]

print(f"Datos cargados: T = {excesos.shape[0]} años, N = {excesos.shape[1]} carteras")

# ========================================================================
# CONTRASTE 1: TEST GRS - LIMITACIÓN POR T < N
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE 1: TEST GRS (LIMITACIÓN POR T < N EN DATOS ANUALES)")
print("=" * 70)

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

print(f"T (años disponibles): {T_grs}")
print(f"N (carteras):         {N_grs}")
print(f"T - N - 1:            {T_grs - N_grs - 1}")

resumen_grs = pd.DataFrame()

if T_grs > N_grs + 1:
    alpha = primera["alpha"].values.reshape(-1, 1)
    Sigma = residuos_grs.cov().values
    mkt_grs = mkt_rf.loc[residuos_grs.index]
    SR_M_squared = (mkt_grs.mean() / mkt_grs.std()) ** 2
    Sigma_inv = np.linalg.inv(Sigma)
    alpha_quad = (alpha.T @ Sigma_inv @ alpha).item()
    grs_stat = ((T_grs - N_grs - 1) / N_grs) * alpha_quad / (1 + SR_M_squared)
    grs_pvalue = 1 - stats.f.cdf(grs_stat, dfn=N_grs, dfd=T_grs - N_grs - 1)
    print(f"\nEstadístico GRS: {grs_stat:.4f}   p-valor: {grs_pvalue:.6f}")
    resumen_grs = pd.DataFrame({
        "estadistico": [grs_stat],
        "p_valor": [grs_pvalue],
        "T": [T_grs], "N": [N_grs],
    })
else:
    print(f"\n→ Test GRS NO APLICABLE: con N={N_grs} y T={T_grs}, no se cumple")
    print(f"  la condición T > N + 1 necesaria para invertir la matriz Sigma")
    print(f"  de covarianzas residuales (N x N). El GRS pleno requiere T >= N+1.")
    print(f"  Esta limitación es estructural del análisis anual y queda")
    print(f"  documentada como tal: el contraste GRS se reporta en la versión")
    print(f"  mensual del trabajo, donde T = 754 >> N = 100.")
    resumen_grs = pd.DataFrame({"observacion": ["no aplicable: T < N+1"]})

# ========================================================================
# CONTRASTE 2: TEST DE WHITE
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE 2: HETEROCEDASTICIDAD DE LA 1A ETAPA ESTATICA (WHITE)")
print("=" * 70)

X_t_w = sm.add_constant(mkt_rf)
resultados_hetero = []

for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t_w], axis=1).dropna()
    Xc = datos[["const", "Mkt-RF"]]
    yc = datos[cartera]
    modelo = sm.OLS(yc, Xc).fit()
    w_lm, w_lm_p, _, _ = smdiag.het_white(modelo.resid, Xc)
    resultados_hetero.append({
        "cartera":       cartera,
        "White_LM":      w_lm,
        "White_p":       w_lm_p,
        "White_rechaza": int(w_lm_p < 0.05),
    })

hetero_df = pd.DataFrame(resultados_hetero).set_index("cartera")
n_white_5 = hetero_df["White_rechaza"].sum()
n_white_1 = (hetero_df["White_p"] < 0.01).sum()

print(f"\nResultados sobre 100 carteras:")
print(f"  Rechazan H0 al 5%:   {n_white_5}/100")
print(f"  Rechazan H0 al 1%:   {n_white_1}/100")
print(f"  p-valor medio:        {hetero_df['White_p'].mean():.4f}")
print(f"  p-valor mediano:      {hetero_df['White_p'].median():.4f}")

if n_white_5 > 50:
    print(f"\n→ Más de la mitad rechazan homocedasticidad al 5%, justificando")
    print(f"  el uso de la metodología con ventana móvil.")

# ========================================================================
# CONTRASTE 3: LINEALIDAD
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE 3: LINEALIDAD (Fama-MacBeth con beta y beta^2)")
print("=" * 70)

gammas_lin = []

for anio_t in betas_rolling.index:
    y      = excesos.loc[anio_t]
    beta_t = betas_rolling.loc[anio_t]
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
    gammas_lin.append(modelo.params.values)

gammas_lin_df = pd.DataFrame(gammas_lin, columns=["gamma_0", "gamma_1", "gamma_2"])
T_lin = len(gammas_lin_df)
medias  = gammas_lin_df.mean()
se      = gammas_lin_df.std() / np.sqrt(T_lin)
t_stats = medias / se
p_vals  = pd.Series(2 * (1 - stats.t.cdf(t_stats.abs(), df=T_lin-1)), index=t_stats.index)

tabla_lin = pd.DataFrame({
    "media":         medias,
    "media_anual_%": medias * 100,
    "error_std":     se,
    "t_stat":        t_stats,
    "p_valor":       p_vals,
})
print("\n" + tabla_lin.round(5).to_string())

# ========================================================================
# CONTRASTE 4: RIESGO IDIOSINCRÁTICO
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE 4: RIESGO IDIOSINCRÁTICO")
print("=" * 70)

gammas_idio = []

for anio_t in betas_rolling.index:
    y       = excesos.loc[anio_t]
    beta_t  = betas_rolling.loc[anio_t]
    sigma_t = sigma_resid_rolling.loc[anio_t]
    X = pd.concat([
        pd.Series(1.0, index=beta_t.index, name="const"),
        beta_t.rename("beta"),
        sigma_t.rename("sigma_resid"),
    ], axis=1)
    datos = pd.concat([y, X], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_idio.append(modelo.params.values)

gammas_idio_df = pd.DataFrame(gammas_idio, columns=["gamma_0", "gamma_1", "gamma_3"])
T_idio = len(gammas_idio_df)
medias  = gammas_idio_df.mean()
se      = gammas_idio_df.std() / np.sqrt(T_idio)
t_stats = medias / se
p_vals  = pd.Series(2 * (1 - stats.t.cdf(t_stats.abs(), df=T_idio-1)), index=t_stats.index)

tabla_idio = pd.DataFrame({
    "media":         medias,
    "media_anual_%": medias * 100,
    "error_std":     se,
    "t_stat":        t_stats,
    "p_valor":       p_vals,
})
print("\n" + tabla_idio.round(5).to_string())

# ========================================================================
# RESUMEN
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN DE LOS CONTRASTES (DATOS ANUALES)")
print("=" * 70)
print(f"{'Contraste':<40} {'Estadístico':>12} {'p-valor':>10}")
print("-" * 70)
print(f"{'GRS':<40} {'no aplicable (T<N+1)':>22}")
print(f"{'Heterocedasticidad White (% rechazan 5%)':<40} "
      f"{n_white_5:>10}/100")
print(f"{'Linealidad (gamma_2 = 0)':<40} {tabla_lin.loc['gamma_2','t_stat']:>12.3f} "
      f"{tabla_lin.loc['gamma_2','p_valor']:>10.4f}")
print(f"{'Riesgo idiosincrático (gamma_3 = 0)':<40} {tabla_idio.loc['gamma_3','t_stat']:>12.3f} "
      f"{tabla_idio.loc['gamma_3','p_valor']:>10.4f}")

# ========================================================================
# GUARDADO
# ========================================================================
tabla_lin.to_csv(os.path.join(CARPETA_RESULTADOS, "contraste_linealidad_anual.csv"))
tabla_idio.to_csv(os.path.join(CARPETA_RESULTADOS, "contraste_idiosincratico_anual.csv"))
hetero_df.to_csv(os.path.join(CARPETA_RESULTADOS, "test_heterocedasticidad_anual.csv"))
resumen_grs.to_csv(os.path.join(CARPETA_RESULTADOS, "test_grs_anual.csv"), index=False)

print(f"\n✓ Tablas exportadas a '{CARPETA_RESULTADOS}/'")
print("✓ Listos para ejecutar paso4_anual_graficos.py")
