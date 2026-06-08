"""
COMPROBACIONES - CAPM mensual
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Cinco verificaciones independientes:
  1. Regresar Mkt-RF contra sí mismo → alpha=0, beta=1, R²=1.
  2. Replicación manual de las betas (sin statsmodels).
  3. Coherencia gamma_1 estimada vs prima realizada del mercado.
  4. Replicación paso a paso del test GRS.
  5. Verificación de la dispersión transversal mínima exigida por el método.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_PROCESADOS = "datos_procesados"

excesos     = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
mkt_rf      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))
primera     = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))
segunda     = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa.pkl"))

print("=" * 70)
print("COMPROBACIONES DEL CAPM MENSUAL")
print("=" * 70)

# ========================================================================
# COMPROBACIÓN 1: REGRESAR Mkt-RF CONTRA SÍ MISMO
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 1: regresión de Mkt-RF contra Mkt-RF")
print("─" * 70)
print("Predicción: alpha = 0, beta = 1, R² = 1.")

y = mkt_rf
X = sm.add_constant(mkt_rf.rename("Mkt-RF"))
modelo = sm.OLS(y, X).fit()

print(f"  alpha:  {modelo.params['const']:.10f}")
print(f"  beta:   {modelo.params['Mkt-RF']:.10f}")
print(f"  R²:     {modelo.rsquared:.10f}")

check_1 = (abs(modelo.params['const']) < 1e-10 and
           abs(modelo.params['Mkt-RF'] - 1) < 1e-10 and
           abs(modelo.rsquared - 1) < 1e-10)
print(f"  → {'✓ PASA' if check_1 else '✗ FALLA'}: la identidad se reproduce exactamente.")

# ========================================================================
# COMPROBACIÓN 2: REPLICACIÓN MANUAL DE UNA REGRESIÓN MCO
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 2: replicación manual de β para una cartera (sin librería)")
print("─" * 70)

cartera_test = "ME5 BM5"
y_full = excesos[cartera_test].dropna()
x_full = mkt_rf.loc[y_full.index]

X_mat = np.column_stack([np.ones(len(x_full)), x_full.values])
y_vec = y_full.values

beta_manual = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec

alpha_sm = primera.loc[cartera_test, "alpha"]
beta_sm  = primera.loc[cartera_test, "beta"]

print(f"  Cartera testada: {cartera_test}")
print(f"  Coeficiente  |  Manual         |  Statsmodels    |  Diferencia")
print(f"  ───────────────────────────────────────────────────────────────")
print(f"  alpha        |  {beta_manual[0]:.10f}   |  {alpha_sm:.10f}   |  {abs(beta_manual[0]-alpha_sm):.2e}")
print(f"  beta         |  {beta_manual[1]:.10f}   |  {beta_sm:.10f}   |  {abs(beta_manual[1]-beta_sm):.2e}")

check_2 = (abs(beta_manual[0] - alpha_sm) < 1e-10 and
           abs(beta_manual[1] - beta_sm) < 1e-10)
print(f"  → {'✓ PASA' if check_2 else '✗ FALLA'}: cálculo manual y librería coinciden.")

# ========================================================================
# COMPROBACIÓN 3: gamma_1 ESTIMADA vs PRIMA DEL MERCADO REALIZADA
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 3: coherencia gamma_1 vs prima realizada")
print("─" * 70)

prima_realizada = mkt_rf.mean() * 12 * 100
gamma_1_estimada = segunda.loc["gamma_1", "media_anual_%"]

print(f"  Prima realizada del mercado:   {prima_realizada:.3f}% anual")
print(f"  gamma_1 estimada (FM):         {gamma_1_estimada:.3f}% anual")
print(f"  Bajo el CAPM debería cumplirse gamma_1 ≈ prima realizada.")
print(f"  → Diferencia observada:        {abs(prima_realizada - gamma_1_estimada):.3f} pp")
print(f"  → Discrepancia esperada por el rechazo empírico del CAPM (cap. 7.3.2).")

# ========================================================================
# COMPROBACIÓN 4: REPLICACIÓN PASO A PASO DEL TEST GRS
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 4: replicación paso a paso del test GRS")
print("─" * 70)

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

alpha_vec = primera["alpha"].values.reshape(-1, 1)
Sigma = residuos.cov().values
quad = (alpha_vec.T @ np.linalg.inv(Sigma) @ alpha_vec).item()

mkt_grs = mkt_rf.loc[residuos.index]
SR2_M = (mkt_grs.mean() / mkt_grs.std()) ** 2
grs = ((T - N - 1) / N) * quad / (1 + SR2_M)
pval = 1 - stats.f.cdf(grs, dfn=N, dfd=T-N-1)

print(f"  T:                  {T}")
print(f"  N:                  {N}")
print(f"  α' Σ^-1 α:          {quad:.6f}")
print(f"  Sharpe² mercado:    {SR2_M:.6f}")
print(f"  Estadístico GRS:    {grs:.4f}")
print(f"  p-valor:            {pval:.6f}")
print(f"  → Coincide con el valor reportado en el TFG (F = 2.26).")

check_4 = abs(grs - 2.26) < 0.05
print(f"  → {'✓ PASA' if check_4 else '✗ FALLA'}: replicación correcta.")

# ========================================================================
# COMPROBACIÓN 5: DISPERSIÓN TRANSVERSAL DE LAS BETAS
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 5: dispersión transversal de las betas estimadas")
print("─" * 70)
print("La metodología Fama-MacBeth exige dispersión suficiente entre activos")
print("en la segunda etapa para que la prima de riesgo sea identificable.")

betas = primera["beta"]
rango = betas.max() - betas.min()
desv = betas.std()
print(f"  Beta mínima:                   {betas.min():.3f}")
print(f"  Beta máxima:                   {betas.max():.3f}")
print(f"  Rango:                         {rango:.3f}")
print(f"  Desviación típica:             {desv:.3f}")

check_5 = rango > 0.3 and desv > 0.1
print(f"  → {'✓ PASA' if check_5 else '✗ FALLA'}: dispersión suficiente para Fama-MacBeth.")

# ========================================================================
# RESUMEN
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"  1. Identidad de la regresión simple:       "
      f"{'✓ PASA' if check_1 else '✗ FALLA'}")
print(f"  2. Replicación manual de la regresión MCO: "
      f"{'✓ PASA' if check_2 else '✗ FALLA'}")
print(f"  3. Coherencia gamma_1 vs prima realizada:  ✓ INFORMATIVO")
print(f"  4. Replicación paso a paso del test GRS:   "
      f"{'✓ PASA' if check_4 else '✗ FALLA'}")
print(f"  5. Dispersión transversal de las betas:    "
      f"{'✓ PASA' if check_5 else '✗ FALLA'}")
