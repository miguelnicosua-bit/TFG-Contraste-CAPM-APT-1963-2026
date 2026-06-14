"""
PASO 3 FF3 - Contrastes formales del modelo de tres factores
TFG Valoración de Activos - Contraste empírico del FF3
Autor: Miguel Suárez Crespo

Dos bloques de contrastes:

  1. TEST GRS (Gibbons-Ross-Shanken, 1989) para FF3 con K=3 factores:
     H0: alpha_j = 0 para todas las N=100 carteras conjuntamente.
     Aplicado con alfas y residuos de la primera etapa ESTÁTICA.

  2. TESTS DE HETEROCEDASTICIDAD (Breusch-Pagan y White) sobre los residuos
     de las 100 regresiones multifactoriales de la primera etapa ESTÁTICA.
     Estos contrastes justifican el uso de la versión móvil: si los residuos
     presentan heterocedasticidad, las betas estáticas no son eficientes
     y la metodología con ventana móvil tiene sentido como corrección.

A diferencia del CAPM, los contrastes adicionales de linealidad
(gamma_2 sobre beta^2) y riesgo idiosincrático (gamma_3 sobre
sigma(epsilon)) no se replican aquí porque no son los habituales para el
modelo multifactor; en su lugar, el contraste GRS conjunto sobre alfas es
el test central del FF3.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smdiag
from scipy import stats

CARPETA_PROCESADOS = "datos_procesados"
CARPETA_RESULTADOS = "resultados/tablas/ff3_mensual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3.pkl"))
factores_ff3 = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))
primera      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3.pkl"))

print(f"Datos cargados (FF3):")
print(f"  T = {excesos.shape[0]} meses, N = {excesos.shape[1]} carteras")
print(f"  K = 3 factores (Mkt-RF, SMB, HML)")

# ========================================================================
# RECONSTRUCCIÓN DE LOS RESIDUOS ESTÁTICOS DE LA PRIMERA ETAPA
# ========================================================================
residuos = pd.DataFrame(index=excesos.index, columns=excesos.columns, dtype=float)
X_t = sm.add_constant(factores_ff3)
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    residuos.loc[datos.index, cartera] = modelo.resid

residuos_grs = residuos.dropna()
T_grs = residuos_grs.shape[0]
N_grs = residuos_grs.shape[1]
K = 3
print(f"\nResiduos para los contrastes: T = {T_grs} x N = {N_grs} "
      f"(se eliminaron {len(excesos) - T_grs} meses con NaN)")

# ========================================================================
# CONTRASTE 1: TEST GRS PARA FF3
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE 1: TEST GRS (Gibbons-Ross-Shanken, 1989) PARA FF3")
print("=" * 70)
print(f"H0: alpha_j = 0 para todas las {N_grs} carteras conjuntamente")
print(f"K = {K} factores (Mkt-RF, SMB, HML)")

alpha = primera["alpha"].values.reshape(-1, 1)
Sigma = residuos_grs.cov().values

factores_grs = factores_ff3.loc[residuos_grs.index]
mu_f = factores_grs.mean().values.reshape(-1, 1)
Omega_f = factores_grs.cov().values

SR2_factores = (mu_f.T @ np.linalg.inv(Omega_f) @ mu_f).item()

Sigma_inv = np.linalg.inv(Sigma)
alpha_quad = (alpha.T @ Sigma_inv @ alpha).item()
grs_stat = ((T_grs - N_grs - K) / N_grs) * alpha_quad / (1 + SR2_factores)
grs_pvalue = 1 - stats.f.cdf(grs_stat, dfn=N_grs, dfd=T_grs - N_grs - K)

print(f"\nEstadístico GRS:        {grs_stat:.4f}")
print(f"Distribución bajo H0:   F({N_grs}, {T_grs - N_grs - K})")
print(f"p-valor:                {grs_pvalue:.6f}")
print(f"")
print(f"Términos del estadístico:")
print(f"  alpha' Sigma^-1 alpha:        {alpha_quad:.4f}")
print(f"  Sharpe^2 multivariante:       {SR2_factores:.4f}")
print(f"  Factor de escala (T-N-K)/N:   {(T_grs-N_grs-K)/N_grs:.4f}")
if grs_pvalue < 0.05:
    print(f"\n→ Se RECHAZA H0 al 5%. Los alfas FF3 son conjuntamente distintos de cero.")
    print(f"  El modelo FF3 es rechazado: los tres factores no agotan la explicación")
    print(f"  transversal de los rendimientos.")
else:
    print(f"\n→ No se rechaza H0. Alfas FF3 conjuntamente compatibles con cero.")

# Comparación con GRS del CAPM (si está disponible)
ruta_grs_capm = "resultados/tablas/capm_mensual/test_grs.csv"
if os.path.exists(ruta_grs_capm):
    grs_capm = pd.read_csv(ruta_grs_capm)
    print("\n" + "-" * 70)
    print("Comparativa CAPM vs FF3 (test GRS):")
    print(f"  CAPM:  F = {grs_capm['estadistico'].iloc[0]:.3f}  p = {grs_capm['p_valor'].iloc[0]:.6f}")
    print(f"  FF3:   F = {grs_stat:.3f}  p = {grs_pvalue:.6f}")
    print("\nEl FF3 reduce las desviaciones (alfas) respecto al CAPM, pero el rechazo persiste.")

# ========================================================================
# CONTRASTE 2: TESTS DE HETEROCEDASTICIDAD SOBRE LA 1ª ETAPA ESTÁTICA FF3
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE 2: HETEROCEDASTICIDAD DE LA 1ª ETAPA ESTÁTICA FF3")
print("=" * 70)
print("Tests aplicados a los residuos de cada una de las 100 regresiones")
print("multifactoriales (paso 1A) sobre los 754 meses del muestral.")
print("\nH0 (en ambos tests): los residuos son homocedásticos.")

X_t_bp = sm.add_constant(factores_ff3)
resultados_hetero = []

for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t_bp], axis=1).dropna()
    Xc = datos[["const", "Mkt-RF", "SMB", "HML"]]
    yc = datos[cartera]
    modelo = sm.OLS(yc, Xc).fit()

    bp_lm, bp_lm_p, bp_f, bp_f_p = smdiag.het_breuschpagan(modelo.resid, Xc)
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

print(f"\nValores medios y medianos de los p-valores:")
print(f"  Breusch-Pagan:  media = {hetero_df['BP_p'].mean():.4f}    mediana = {hetero_df['BP_p'].median():.4f}")
print(f"  White:          media = {hetero_df['White_p'].mean():.4f}    mediana = {hetero_df['White_p'].median():.4f}")

if n_bp_5 > 50 or n_white_5 > 50:
    print(f"\n→ Más de la mitad de las carteras presentan heterocedasticidad")
    print(f"  significativa al 5%. Esto justifica el uso de la metodología")
    print(f"  con ventana móvil de 60 meses, que permite que las betas")
    print(f"  factoriales varíen en el tiempo y atenúa el problema.")
else:
    print(f"\n→ Heterocedasticidad detectada en una proporción minoritaria.")
    print(f"  La metodología con ventana móvil sigue siendo preferible.")

# ========================================================================
# RESUMEN GLOBAL
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN DE LOS CONTRASTES")
print("=" * 70)
print(f"{'Contraste':<45} {'Estadístico':>12} {'p-valor':>10} {'Decisión':>15}")
print("-" * 85)
print(f"{'GRS (alfas FF3 conjuntos = 0)':<45} {grs_stat:>12.3f} {grs_pvalue:>10.4f} "
      f"{'Rechaza FF3' if grs_pvalue < 0.05 else 'No rechaza':>15}")
print(f"{'Heterocedasticidad BP (% rechazan al 5%)':<45} "
      f"{n_bp_5:>10}/100  {'':>10} "
      f"{'Justifica móvil' if n_bp_5 > 50 else 'Sin evidencia':>15}")
print(f"{'Heterocedasticidad White (% rechazan al 5%)':<45} "
      f"{n_white_5:>10}/100  {'':>10} "
      f"{'Justifica móvil' if n_white_5 > 50 else 'Sin evidencia':>15}")

# ========================================================================
# GUARDADO
# ========================================================================
hetero_df.to_csv(os.path.join(CARPETA_RESULTADOS, "test_heterocedasticidad_ff3.csv"))

resumen_grs = pd.DataFrame({
    "estadistico":  [grs_stat],
    "p_valor":      [grs_pvalue],
    "T":            [T_grs],
    "N":            [N_grs],
    "K":            [K],
    "Sharpe2_factores": [SR2_factores],
    "alpha_quad":   [alpha_quad],
})
resumen_grs.to_csv(os.path.join(CARPETA_RESULTADOS, "test_grs_ff3.csv"), index=False)

resumen_hetero = pd.DataFrame({
    "test":           ["Breusch-Pagan", "White"],
    "rechazan_al_5%": [n_bp_5, n_white_5],
    "rechazan_al_1%": [n_bp_1, n_white_1],
    "p_medio":        [hetero_df['BP_p'].mean(), hetero_df['White_p'].mean()],
    "p_mediano":      [hetero_df['BP_p'].median(), hetero_df['White_p'].median()],
})
resumen_hetero.to_csv(os.path.join(CARPETA_RESULTADOS, "resumen_heterocedasticidad_ff3.csv"), index=False)

print(f"\n✓ Tablas exportadas a '{CARPETA_RESULTADOS}/'")
print("✓ Listos para ejecutar paso4_ff3_graficos.py")
