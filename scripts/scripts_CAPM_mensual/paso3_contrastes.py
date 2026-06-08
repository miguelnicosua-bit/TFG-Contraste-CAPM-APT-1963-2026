"""
PASO 3 - Contrastes formales del CAPM
TFG Valoración de Activos - Contraste empírico del CAPM
Autor: Miguel Suárez Crespo

Tres contrastes:
  1. Test GRS (Gibbons-Ross-Shanken, 1989): H0: alpha_i = 0 para todas las carteras.
  2. Contraste de linealidad: añadir beta^2 a la sección cruzada.
  3. Contraste del riesgo idiosincrático: añadir sigma(epsilon_i) a la sección cruzada.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
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

print(f"Datos cargados: T = {excesos.shape[0]} meses, N = {excesos.shape[1]} carteras")

# ========================================================================
# CONTRASTE 1: TEST GRS
# ========================================================================
# Necesitamos:
#   - los alfas estimados (vector N x 1)         -> ya están en primera
#   - la matriz de covarianzas de los residuos (N x N)
# Para el GRS necesitamos una matriz T x N de residuos SIN NaN.
# Estrategia: eliminamos los meses con algún NaN en cualquier cartera.

# Recalculamos los residuos de la primera etapa
residuos = pd.DataFrame(index=excesos.index, columns=excesos.columns, dtype=float)

X_t = sm.add_constant(mkt_rf)
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    residuos.loc[datos.index, cartera] = modelo.resid

# Para el GRS necesitamos residuos sin NaN. Eliminamos meses problemáticos.
residuos_grs = residuos.dropna()
T_grs = residuos_grs.shape[0]
N_grs = residuos_grs.shape[1]
print(f"\nGRS: usando T = {T_grs} meses y N = {N_grs} carteras "
      f"(se eliminaron {len(excesos) - T_grs} meses con NaN)")

alpha = primera["alpha"].values.reshape(-1, 1)          # N x 1
Sigma = residuos_grs.cov().values                       # N x N (matriz de cov. residual)

# Sharpe ratio al cuadrado del mercado en el mismo período
mkt_grs = mkt_rf.loc[residuos_grs.index]
SR_M_squared = (mkt_grs.mean() / mkt_grs.std()) ** 2

# Estadístico GRS
Sigma_inv = np.linalg.inv(Sigma)
alpha_quad = (alpha.T @ Sigma_inv @ alpha).item()

grs_stat = ((T_grs - N_grs - 1) / N_grs) * alpha_quad / (1 + SR_M_squared)
grs_pvalue = 1 - stats.f.cdf(grs_stat, dfn=N_grs, dfd=T_grs - N_grs - 1)

print("\n" + "=" * 70)
print("CONTRASTE 1: TEST GRS (Gibbons-Ross-Shanken, 1989)")
print("=" * 70)
print(f"H0: alpha_i = 0 para todas las {N_grs} carteras conjuntamente")
print(f"")
print(f"Estadístico GRS:   {grs_stat:.4f}")
print(f"Distribución bajo H0: F({N_grs}, {T_grs - N_grs - 1})")
print(f"p-valor:           {grs_pvalue:.6f}")
print(f"Sharpe^2 mercado:  {SR_M_squared:.4f}")
print(f"alpha' Sigma^-1 alpha: {alpha_quad:.4f}")
if grs_pvalue < 0.05:
    print(f"\n→ Se RECHAZA H0 al 5%. Los alfas son CONJUNTAMENTE distintos de cero.")
    print(f"  El CAPM es rechazado: las 100 carteras presentan precios incompatibles")
    print(f"  con la prediccion del modelo.")
else:
    print(f"\n→ No se rechaza H0. Los alfas conjuntamente compatibles con cero.")

# ========================================================================
# CONTRASTE 2: LINEALIDAD (añadir beta^2 a la sección cruzada)
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE 2: LINEALIDAD (Fama-MacBeth con beta y beta^2)")
print("=" * 70)
print("H0: gamma_2 = 0 en la regresión:")
print("    r_{i,t} - r_f,t = gamma_0,t + gamma_1,t beta_i + gamma_2,t beta_i^2 + u")

betas = primera["beta"]
betas_sq = betas ** 2

# Regresores: constante + beta + beta^2
X_lin = pd.DataFrame({
    "const": 1.0,
    "beta": betas.values,
    "beta_sq": betas_sq.values
}, index=betas.index)

gammas_lin = []
for fecha in excesos.index:
    y = excesos.loc[fecha]
    datos = pd.concat([y, X_lin], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_lin.append(modelo.params.values)

gammas_lin = pd.DataFrame(gammas_lin, columns=["gamma_0", "gamma_1", "gamma_2"])

T = len(gammas_lin)
medias = gammas_lin.mean()
se = gammas_lin.std() / np.sqrt(T)
t_stats = medias / se
p_vals = pd.Series(2 * (1 - stats.t.cdf(t_stats.abs(), df=T-1)), index=t_stats.index)

tabla_lin = pd.DataFrame({
    "media": medias,
    "media_anual_%": medias * 12 * 100,
    "error_std": se,
    "t_stat": t_stats,
    "p_valor": p_vals,
})
print("\n" + tabla_lin.round(5).to_string())

g2_p = p_vals["gamma_2"]
print(f"\nContraste H0: gamma_2 = 0  (p-valor = {g2_p:.4f})")
if g2_p < 0.05:
    print(f"→ Se RECHAZA H0 al 5%. La relación rendimiento-beta NO es lineal.")
    print(f"  El CAPM es rechazado por incumplir la linealidad.")
else:
    print(f"→ No se rechaza H0. Compatible con linealidad.")

# ========================================================================
# CONTRASTE 3: RIESGO IDIOSINCRÁTICO (añadir sigma_residual)
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE 3: RIESGO IDIOSINCRÁTICO")
print("=" * 70)
print("H0: gamma_3 = 0 en la regresión:")
print("    r_{i,t} - r_f,t = gamma_0,t + gamma_1,t beta_i + gamma_3,t sigma(e_i) + u")

sigma_resid = primera["sigma_residual"]

X_idio = pd.DataFrame({
    "const": 1.0,
    "beta": betas.values,
    "sigma_resid": sigma_resid.values
}, index=betas.index)

gammas_idio = []
for fecha in excesos.index:
    y = excesos.loc[fecha]
    datos = pd.concat([y, X_idio], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_idio.append(modelo.params.values)

gammas_idio = pd.DataFrame(gammas_idio, columns=["gamma_0", "gamma_1", "gamma_3"])

T = len(gammas_idio)
medias = gammas_idio.mean()
se = gammas_idio.std() / np.sqrt(T)
t_stats = medias / se
p_vals = pd.Series(2 * (1 - stats.t.cdf(t_stats.abs(), df=T-1)), index=t_stats.index)

tabla_idio = pd.DataFrame({
    "media": medias,
    "media_anual_%": medias * 12 * 100,
    "error_std": se,
    "t_stat": t_stats,
    "p_valor": p_vals,
})
print("\n" + tabla_idio.round(5).to_string())

g3_p = p_vals["gamma_3"]
print(f"\nContraste H0: gamma_3 = 0  (p-valor = {g3_p:.4f})")
if g3_p < 0.05:
    print(f"→ Se RECHAZA H0 al 5%. El riesgo idiosincrático SÍ está valorado.")
    print(f"  El CAPM es rechazado: el mercado compensa riesgo diversificable.")
else:
    print(f"→ No se rechaza H0. El riesgo idiosincrático no parece tener precio.")

# ========================================================================
# RESUMEN GLOBAL
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN DE LOS TRES CONTRASTES")
print("=" * 70)
print(f"{'Contraste':<35} {'Estadístico':>12} {'p-valor':>10} {'Conclusión':>15}")
print("-" * 75)
print(f"{'GRS (alfas conjuntos = 0)':<35} {grs_stat:>12.3f} {grs_pvalue:>10.4f} "
      f"{'Rechaza CAPM' if grs_pvalue < 0.05 else 'No rechaza':>15}")
print(f"{'Linealidad (gamma_2 = 0)':<35} {tabla_lin.loc['gamma_2','t_stat']:>12.3f} "
      f"{tabla_lin.loc['gamma_2','p_valor']:>10.4f} "
      f"{'Rechaza CAPM' if tabla_lin.loc['gamma_2','p_valor'] < 0.05 else 'No rechaza':>15}")
print(f"{'Riesgo idiosincrático (gamma_3=0)':<35} {tabla_idio.loc['gamma_3','t_stat']:>12.3f} "
      f"{tabla_idio.loc['gamma_3','p_valor']:>10.4f} "
      f"{'Rechaza CAPM' if tabla_idio.loc['gamma_3','p_valor'] < 0.05 else 'No rechaza':>15}")

# ========================================================================
# GUARDADO
# ========================================================================
tabla_lin.to_csv(os.path.join(CARPETA_RESULTADOS, "contraste_linealidad.csv"))
tabla_idio.to_csv(os.path.join(CARPETA_RESULTADOS, "contraste_idiosincratico.csv"))

resumen_grs = pd.DataFrame({
    "estadistico": [grs_stat],
    "p_valor": [grs_pvalue],
    "T": [T_grs], "N": [N_grs],
    "Sharpe2_mkt": [SR_M_squared],
    "alpha_quad": [alpha_quad],
})
resumen_grs.to_csv(os.path.join(CARPETA_RESULTADOS, "test_grs.csv"), index=False)

print(f"\n✓ Tablas exportadas a '{CARPETA_RESULTADOS}/'")
print("✓ Listos para ejecutar paso4_graficos.py")
