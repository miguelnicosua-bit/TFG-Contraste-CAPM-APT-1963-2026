"""
PASO 3 ANUAL - Contrastes formales del CAPM con datos anuales
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Tres contrastes complementarios:
  1. Linealidad de la SML: añadir beta^2 como regresor adicional.
  2. Riesgo idiosincrático: añadir sigma(epsilon) como regresor adicional.
  3. Test GRS conjunto de los alfas.

⚠ NOTA TÉCNICA IMPORTANTE: Como T = 62 < N = 100, la matriz de covarianzas
de los residuos Σ (de dimensión 100x100) es necesariamente SINGULAR y no
puede invertirse para el test GRS clásico. Este paso ofrece tres alternativas:
  (a) Reportar la imposibilidad del GRS estándar.
  (b) Aplicar el GRS con la pseudo-inversa de Moore-Penrose como aproximación.
  (c) Aplicar el GRS sobre un subconjunto reducido de carteras (25 carteras
      agregando los deciles en quintiles).
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_ANUAL = "datos_procesados_anual"
CARPETA_RESULTADOS = "resultados/tablas/capm_anual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos     = pd.read_pickle(os.path.join(CARPETA_ANUAL, "excesos_carteras_anual.pkl"))
mkt_rf      = pd.read_pickle(os.path.join(CARPETA_ANUAL, "mkt_rf_anual.pkl"))
primera_a   = pd.read_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_anual.pkl"))

betas = primera_a["beta"]
betas_sq = betas ** 2
sigma_eps = primera_a["sigma_residual"]

print("=" * 70)
print("PASO 3 ANUAL - CONTRASTES FORMALES DEL CAPM")
print("=" * 70)

# ========================================================================
# CONTRASTE 1: LINEALIDAD DE LA SML (beta^2 como regresor adicional)
# ========================================================================
print("\n" + "─" * 70)
print("CONTRASTE 1: LINEALIDAD (gamma_2 asociado a beta^2)")
print("─" * 70)

X_lin = sm.add_constant(pd.concat([
    betas.rename("beta"),
    betas_sq.rename("beta_sq")
], axis=1))

gammas_lin = []
for anio in excesos.index:
    y = excesos.loc[anio]
    datos = pd.concat([y, X_lin], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_lin.append(modelo.params.values)

gammas_lin = pd.DataFrame(gammas_lin, columns=["g0", "g1", "g2"])
T_lin = len(gammas_lin)

g2_mean   = gammas_lin["g2"].mean()
g2_se     = gammas_lin["g2"].std() / np.sqrt(T_lin)
g2_t      = g2_mean / g2_se
g2_p      = 2 * (1 - stats.t.cdf(abs(g2_t), df=T_lin-1))

print(f"  gamma_2 (coef. de beta^2) anualizado:   {g2_mean * 100:.3f}%")
print(f"  t-estadístico:                          {g2_t:.3f}")
print(f"  p-valor:                                {g2_p:.4f}")
print(f"  → {'RECHAZA' if g2_p < 0.05 else 'NO rechaza'} la linealidad de la SML al 5%.")

# ========================================================================
# CONTRASTE 2: RIESGO IDIOSINCRÁTICO (sigma(epsilon) como regresor adicional)
# ========================================================================
print("\n" + "─" * 70)
print("CONTRASTE 2: RIESGO IDIOSINCRÁTICO (gamma_3 asociado a sigma_eps)")
print("─" * 70)

X_idio = sm.add_constant(pd.concat([
    betas.rename("beta"),
    sigma_eps.rename("sigma_eps")
], axis=1))

gammas_idio = []
for anio in excesos.index:
    y = excesos.loc[anio]
    datos = pd.concat([y, X_idio], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_idio.append(modelo.params.values)

gammas_idio = pd.DataFrame(gammas_idio, columns=["g0", "g1", "g3"])
T_idio = len(gammas_idio)

g3_mean   = gammas_idio["g3"].mean()
g3_se     = gammas_idio["g3"].std() / np.sqrt(T_idio)
g3_t      = g3_mean / g3_se
g3_p      = 2 * (1 - stats.t.cdf(abs(g3_t), df=T_idio-1))

print(f"  gamma_3 (coef. de sigma_eps) anualizado: {g3_mean * 100:.3f}%")
print(f"  t-estadístico:                           {g3_t:.3f}")
print(f"  p-valor:                                 {g3_p:.4f}")
print(f"  → {'RECHAZA' if g3_p < 0.05 else 'NO rechaza'} la no remuneración del riesgo "
      f"idiosincrático al 5%.")

# ========================================================================
# CONTRASTE 3: TEST GRS (con caveat técnico)
# ========================================================================
print("\n" + "─" * 70)
print("CONTRASTE 3: TEST GRS DE LOS ALFAS CONJUNTOS")
print("─" * 70)

# Calculamos los residuos de la primera etapa
residuos = pd.DataFrame(index=excesos.index, columns=excesos.columns, dtype=float)
X_t = sm.add_constant(mkt_rf.rename("Mkt-RF"))
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    residuos.loc[datos.index, cartera] = modelo.resid

residuos = residuos.dropna()
T = residuos.shape[0]
N = residuos.shape[1]

print(f"  T (años):                  {T}")
print(f"  N (carteras):              {N}")
print(f"  Como T < N+2, la matriz Σ ({N}x{N}) es SINGULAR.")
print(f"  El test GRS clásico (que requiere invertir Σ) NO es aplicable.")

# Opción (b): GRS con pseudo-inversa de Moore-Penrose como aproximación
print("\n  Aproximación (b): GRS con pseudo-inversa de Moore-Penrose")
print("  " + "─" * 60)

alpha_vec = primera_a["alpha"].values.reshape(-1, 1)
Sigma = residuos.cov().values
Sigma_pinv = np.linalg.pinv(Sigma)

quad = (alpha_vec.T @ Sigma_pinv @ alpha_vec).item()
SR2_M = (mkt_rf.mean() / mkt_rf.std()) ** 2

# Aproximación al GRS (debe interpretarse con MUCHA cautela)
grs_aprox = ((T - N - 1) / N) * quad / (1 + SR2_M)
print(f"  alpha' Σ⁺ alpha (con pseudo-inversa):   {quad:.4f}")
print(f"  Sharpe² del mercado:                    {SR2_M:.4f}")
print(f"  GRS aproximado:                         {grs_aprox:.4f}")
print(f"  ⚠ Este valor NO debe interpretarse formalmente; sirve solo como")
print(f"    indicación cualitativa.")

# Opción (c): GRS sobre las 25 carteras 5x5 (agregando deciles en quintiles)
print("\n  Aproximación (c): GRS sobre las 25 carteras 5x5 (T > N posible)")
print("  " + "─" * 60)

import re
def parsear_nombre(nombre):
    n = nombre.strip()
    if n == "SMALL LoBM": return 1, 1
    if n == "SMALL HiBM": return 1, 10
    if n == "BIG LoBM":   return 10, 1
    if n == "BIG HiBM":   return 10, 10
    m = re.match(r"ME(\d+)\s+BM(\d+)", n)
    if m:
        return int(m.group(1)), int(m.group(2))

# Mapear cada cartera 10x10 a quintil 5x5: 1-2→1, 3-4→2, 5-6→3, 7-8→4, 9-10→5
def decil_a_quintil(d):
    return (d - 1) // 2 + 1

mapping_5x5 = {}
for c in excesos.columns:
    s, b = parsear_nombre(c)
    q_s = decil_a_quintil(s)
    q_b = decil_a_quintil(b)
    mapping_5x5[c] = f"S{q_s}B{q_b}"

# Agregar excesos en 25 carteras (media simple dentro de cada grupo)
excesos_5x5 = excesos.copy()
excesos_5x5.columns = [mapping_5x5[c] for c in excesos.columns]
excesos_5x5 = excesos_5x5.T.groupby(level=0).mean().T

# Re-estimar alfas y residuos para las 25 carteras
alfas_5x5 = []
residuos_5x5 = pd.DataFrame(index=excesos_5x5.index, columns=excesos_5x5.columns, dtype=float)
for cartera in excesos_5x5.columns:
    y = excesos_5x5[cartera]
    datos = pd.concat([y, X_t], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    alfas_5x5.append(modelo.params["const"])
    residuos_5x5.loc[datos.index, cartera] = modelo.resid

residuos_5x5 = residuos_5x5.dropna()
T5 = residuos_5x5.shape[0]
N5 = residuos_5x5.shape[1]
alpha_5x5 = np.array(alfas_5x5).reshape(-1, 1)
Sigma_5x5 = residuos_5x5.cov().values

if T5 > N5 + 1:
    quad_5x5 = (alpha_5x5.T @ np.linalg.inv(Sigma_5x5) @ alpha_5x5).item()
    grs_5x5 = ((T5 - N5 - 1) / N5) * quad_5x5 / (1 + SR2_M)
    p_5x5 = 1 - stats.f.cdf(grs_5x5, dfn=N5, dfd=T5 - N5 - 1)
    print(f"  T (años):                          {T5}")
    print(f"  N (carteras agregadas a 5x5):      {N5}")
    print(f"  alpha' Σ^-1 alpha:                 {quad_5x5:.4f}")
    print(f"  Estadístico GRS (5x5):             {grs_5x5:.4f}")
    print(f"  p-valor F({N5}, {T5-N5-1}):           {p_5x5:.6f}")
    print(f"  → {'RECHAZA' if p_5x5 < 0.05 else 'NO rechaza'} la hipótesis conjunta "
          f"de alfas nulos al 5%.")
else:
    print(f"  Incluso con agregación a 5x5, T={T5} <= N+1={N5+1}. No aplicable.")

# ========================================================================
# RESUMEN
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN DE CONTRASTES (DATOS ANUALES)")
print("=" * 70)

tabla = pd.DataFrame({
    "Contraste": [
        "Linealidad (beta^2)",
        "Riesgo idiosincrático (sigma_eps)",
        "GRS clásico (100 carteras)",
        "GRS aproximado (25 carteras 5x5)" if T5 > N5+1 else "GRS aproximado (n.d.)",
    ],
    "Estadístico": [
        f"t = {g2_t:.3f}",
        f"t = {g3_t:.3f}",
        "N/A (Σ singular)",
        f"F = {grs_5x5:.3f}" if T5 > N5+1 else "N/A",
    ],
    "p-valor": [
        f"{g2_p:.4f}",
        f"{g3_p:.4f}",
        "—",
        f"{p_5x5:.6f}" if T5 > N5+1 else "—",
    ],
    "Decisión al 5%": [
        "Rechaza" if g2_p < 0.05 else "No rechaza",
        "Rechaza" if g3_p < 0.05 else "No rechaza",
        "—",
        ("Rechaza" if p_5x5 < 0.05 else "No rechaza") if T5 > N5+1 else "—",
    ],
})
print(tabla.to_string(index=False))

# ========================================================================
# GUARDADO
# ========================================================================
tabla.to_csv(os.path.join(CARPETA_RESULTADOS, "contrastes_anual.csv"), index=False)

contrastes_dict = {
    "linealidad": {"coef": g2_mean, "t_stat": g2_t, "p_valor": g2_p},
    "idiosincratico": {"coef": g3_mean, "t_stat": g3_t, "p_valor": g3_p},
    "grs_5x5": {"stat": grs_5x5, "p_valor": p_5x5, "T": T5, "N": N5} if T5 > N5+1 else None,
}
pd.to_pickle(contrastes_dict, os.path.join(CARPETA_ANUAL, "contrastes_anual.pkl"))

print(f"\n✓ Resultados guardados en '{CARPETA_ANUAL}/'")
print("✓ Listos para ejecutar paso4_anual_graficos.py")
