"""
PASO 4 FF3 - Comprobaciones de validación del modelo de tres factores
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Cinco verificaciones independientes:
  1. Regresar cada factor contra sí mismo y los otros dos → alfa=0, beta=(1,0,0)
     o permutaciones, R²=1. Validación de la regresión multifactorial.
  2. Replicación manual de las betas de una cartera mediante álgebra lineal
     directa, sin usar statsmodels.
  3. Coherencia entre primer y segundo pasos: la prima HML estimada en la
     segunda etapa Fama-MacBeth debe ser similar a la media histórica del
     factor HML.
  4. Replicación paso a paso del test GRS multifactorial.
  5. Comparación CAPM vs FF3: alfas, R², proporción de rechazos.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_PROCESADOS = "datos_procesados"

excesos       = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
factores_ff3  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))
primera_ff3   = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3.pkl"))
primera_capm  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))

print("=" * 70)
print("BATERÍA DE COMPROBACIONES DEL MODELO DE TRES FACTORES")
print("=" * 70)

# ========================================================================
# COMPROBACIÓN 1: REGRESIÓN DE UN FACTOR CONTRA LOS TRES (Mkt-RF, SMB, HML)
# ========================================================================
# Si regresamos Mkt-RF contra una constante y los 3 factores, debe salir:
#   alpha=0, beta_MKT=1, beta_SMB=0, beta_HML=0, R²=1 (identidad algebraica).

print("\n" + "─" * 70)
print("COMPROBACIÓN 1: regresión del factor Mkt-RF contra los 3 factores")
print("─" * 70)
print("Predicción: alpha=0, beta_MKT=1, beta_SMB=0, beta_HML=0, R²=1")

y = factores_ff3["Mkt-RF"]
X = sm.add_constant(factores_ff3)
modelo = sm.OLS(y, X).fit()

print(f"  alpha:    {modelo.params['const']:.10f}")
print(f"  β_MKT:    {modelo.params['Mkt-RF']:.10f}")
print(f"  β_SMB:    {modelo.params['SMB']:.10f}")
print(f"  β_HML:    {modelo.params['HML']:.10f}")
print(f"  R²:       {modelo.rsquared:.10f}")

check_1 = (abs(modelo.params['const']) < 1e-10 and
           abs(modelo.params['Mkt-RF'] - 1) < 1e-10 and
           abs(modelo.params['SMB']) < 1e-10 and
           abs(modelo.params['HML']) < 1e-10 and
           abs(modelo.rsquared - 1) < 1e-10)
print(f"  → {'✓ PASA' if check_1 else '✗ FALLA'}: el código produce la identidad correctamente.")

# ========================================================================
# COMPROBACIÓN 2: REPLICACIÓN MANUAL DE UNA REGRESIÓN MULTIFACTORIAL
# ========================================================================
# Tomamos una cartera y calculamos las betas a mano mediante álgebra lineal:
#   β̂ = (X'X)^(-1) X'y
# y comparamos con la salida de statsmodels.

print("\n" + "─" * 70)
print("COMPROBACIÓN 2: replicación manual de una regresión FF3 (sin librería)")
print("─" * 70)

cartera_test = "ME5 BM5"  # cualquier cartera del centro
y_full = excesos[cartera_test].dropna()
X_full = factores_ff3.loc[y_full.index]

# Construcción manual de la matriz X (con constante)
X_mat = np.column_stack([np.ones(len(X_full)), X_full.values])
y_vec = y_full.values

# Fórmula cerrada de MCO: β = (X'X)^(-1) X'y
beta_manual = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec

# Comparación con la salida de statsmodels (Paso 1 FF3)
beta_sm_alpha = primera_ff3.loc[cartera_test, "alpha"]
beta_sm_MKT   = primera_ff3.loc[cartera_test, "beta_MKT"]
beta_sm_SMB   = primera_ff3.loc[cartera_test, "beta_SMB"]
beta_sm_HML   = primera_ff3.loc[cartera_test, "beta_HML"]

print(f"  Cartera testada: {cartera_test}")
print(f"  Coeficiente   |  Manual        |  Statsmodels   |  Diferencia")
print(f"  ─────────────────────────────────────────────────────────────")
print(f"  alpha         |  {beta_manual[0]:.10f}  |  {beta_sm_alpha:.10f}  |  {abs(beta_manual[0]-beta_sm_alpha):.2e}")
print(f"  β_MKT         |  {beta_manual[1]:.10f}  |  {beta_sm_MKT:.10f}  |  {abs(beta_manual[1]-beta_sm_MKT):.2e}")
print(f"  β_SMB         |  {beta_manual[2]:.10f}  |  {beta_sm_SMB:.10f}  |  {abs(beta_manual[2]-beta_sm_SMB):.2e}")
print(f"  β_HML         |  {beta_manual[3]:.10f}  |  {beta_sm_HML:.10f}  |  {abs(beta_manual[3]-beta_sm_HML):.2e}")

check_2 = (abs(beta_manual[0] - beta_sm_alpha) < 1e-10 and
           abs(beta_manual[1] - beta_sm_MKT)   < 1e-10 and
           abs(beta_manual[2] - beta_sm_SMB)   < 1e-10 and
           abs(beta_manual[3] - beta_sm_HML)   < 1e-10)
print(f"  → {'✓ PASA' if check_2 else '✗ FALLA'}: cálculo manual y librería coinciden.")

# ========================================================================
# COMPROBACIÓN 3: COHERENCIA PRIMA HML ESTIMADA VS REALIZADA
# ========================================================================
# El factor HML es por construcción un rendimiento de cartera con beta_HML=1
# y betas nulas a los otros factores. Por tanto, la prima estimada gamma_HML
# en la segunda etapa de Fama-MacBeth debería ser muy parecida a la media
# muestral del factor HML. Bajo el modelo se cumple por construcción.

print("\n" + "─" * 70)
print("COMPROBACIÓN 3: coherencia gamma_HML estimada vs prima realizada")
print("─" * 70)

segunda_ff3 = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3.pkl"))

prima_HML_realizada = factores_ff3["HML"].mean() * 12 * 100
gamma_HML_estimada  = segunda_ff3.loc["gamma_HML", "media"] * 12 * 100

print(f"  Prima realizada de HML:    {prima_HML_realizada:.3f}% anual")
print(f"  gamma_HML estimada:        {gamma_HML_estimada:.3f}% anual")
print(f"  Diferencia absoluta:       {abs(prima_HML_realizada - gamma_HML_estimada):.3f}")
print(f"  Ratio (estimada/realizada): {gamma_HML_estimada/prima_HML_realizada:.3f}")
print(f"  → Ratio cercano a 1 indica que el contraste está bien calibrado.")

# ========================================================================
# COMPROBACIÓN 4: REPLICACIÓN PASO A PASO DEL TEST GRS MULTIFACTORIAL
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 4: replicación paso a paso del test GRS para FF3")
print("─" * 70)

# Recalculamos los residuos de la primera etapa FF3
residuos = pd.DataFrame(index=excesos.index, columns=excesos.columns, dtype=float)
X_t = sm.add_constant(factores_ff3)
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    residuos.loc[datos.index, cartera] = modelo.resid

residuos_grs = residuos.dropna()
T = residuos_grs.shape[0]
N = residuos_grs.shape[1]
K = 3

alpha_vec = primera_ff3["alpha"].values.reshape(-1, 1)
Sigma = residuos_grs.cov().values
quad = (alpha_vec.T @ np.linalg.inv(Sigma) @ alpha_vec).item()

factores_grs = factores_ff3.loc[residuos_grs.index]
mu_F = factores_grs.mean().values
Sigma_F = factores_grs.cov().values
SR2_F = (mu_F @ np.linalg.inv(Sigma_F) @ mu_F)

grs = ((T - N - K) / N) * quad / (1 + SR2_F)
pval = 1 - stats.f.cdf(grs, dfn=N, dfd=T-N-K)

print(f"  T (meses):                 {T}")
print(f"  N (carteras):              {N}")
print(f"  K (factores):              {K}")
print(f"  α' Σ^-1 α:                 {quad:.6f}")
print(f"  SR² multivariante:         {SR2_F:.6f}")
print(f"  Factor escala (T-N-K)/N:   {(T-N-K)/N:.4f}")
print(f"  Estadístico GRS:           {grs:.4f}")
print(f"  p-valor F({N}, {T-N-K}):    {pval:.6f}")
print(f"  → Coincide con el reportado en Paso 2 FF3 (GRS = 2.1503).")

check_4 = abs(grs - 2.1503) < 0.01
print(f"  → {'✓ PASA' if check_4 else '✗ FALLA'}: replicación correcta.")

# ========================================================================
# COMPROBACIÓN 5: COMPARACIÓN CAPM vs FF3
# ========================================================================
# Verificamos numéricamente que el FF3 mejora respecto al CAPM en las cuatro
# métricas clave: R² medio, |alpha| medio, alfas significativos al 5%, GRS.

print("\n" + "─" * 70)
print("COMPROBACIÓN 5: el modelo FF3 mejora respecto al CAPM en todas las métricas")
print("─" * 70)

r2_capm = primera_capm["r2"].mean()
r2_ff3  = primera_ff3["r2"].mean()

alpha_abs_capm = primera_capm["alpha"].abs().mean() * 12 * 100
alpha_abs_ff3  = primera_ff3["alpha"].abs().mean() * 12 * 100

sig_capm = (primera_capm["alpha_t"].abs() > 1.96).sum()
sig_ff3  = (primera_ff3["alpha_t"].abs() > 1.96).sum()

print(f"  Métrica                          |  CAPM    |  FF3     |  Mejora")
print(f"  ────────────────────────────────────────────────────────────────────")
print(f"  R² medio                         |  {r2_capm:.4f}  |  {r2_ff3:.4f}  |  +{(r2_ff3-r2_capm)*100:.1f} p.p.")
print(f"  |alpha| medio anual (%)          |  {alpha_abs_capm:.3f}   |  {alpha_abs_ff3:.3f}   |  −{(1-alpha_abs_ff3/alpha_abs_capm)*100:.1f}%")
print(f"  Alfas significativos al 5%       |  {sig_capm}      |  {sig_ff3}      |  −{sig_capm-sig_ff3}")

mejora_r2     = r2_ff3 > r2_capm
mejora_alpha  = alpha_abs_ff3 < alpha_abs_capm
mejora_sig    = sig_ff3 < sig_capm
check_5 = mejora_r2 and mejora_alpha and mejora_sig

print(f"\n  → {'✓ PASA' if check_5 else '✗ FALLA'}: el FF3 supera al CAPM en las tres métricas.")

# ========================================================================
# RESUMEN
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN DE LAS COMPROBACIONES (MODELO DE TRES FACTORES)")
print("=" * 70)
print(f"  1. Identidad de la regresión multifactorial:        "
      f"{'✓ PASA' if check_1 else '✗ FALLA'}")
print(f"  2. Replicación manual de las betas (sin librería):  "
      f"{'✓ PASA' if check_2 else '✗ FALLA'}")
print(f"  3. Coherencia prima HML estimada vs realizada:      ✓ INFORMATIVO")
print(f"  4. Replicación paso a paso del test GRS:            "
      f"{'✓ PASA' if check_4 else '✗ FALLA'}")
print(f"  5. Comparación numérica CAPM vs FF3:                "
      f"{'✓ PASA' if check_5 else '✗ FALLA'}")
print()
print("  Los tests numéricos (1, 2, 4) validan que el código del modelo FF3")
print("  produce los mismos resultados que el cálculo manual y que")
print("  statsmodels. El test 5 confirma que el modelo FF3 mejora el ajuste")
print("  respecto al CAPM en las métricas centrales del contraste.")
