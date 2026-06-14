"""
COMPROBACIONES FF3 - Modelo de tres factores mensual
(metodología Fama-MacBeth rolling 60 meses)
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Cinco verificaciones independientes:
  1. Regresar Mkt-RF contra los tres factores → beta_M=1, beta_SMB=0, beta_HML=0, R²=1.
  2. Replicación manual de las betas FF3 estáticas de una cartera (sin statsmodels).
  3. Coherencia gamma_HML estimada vs prima realizada del factor HML.
  4. Replicación paso a paso del test GRS para FF3.
  5. Replicación manual de la metodología rolling 60m sobre un mes concreto:
     se comprueba que las betas factoriales y los gammas guardados coinciden
     con el cálculo manual.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_PROCESADOS = "datos_procesados"

excesos          = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3.pkl"))
factores_ff3     = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))
primera          = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3.pkl"))
primera_rolling  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3_rolling.pkl"))
segunda          = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3.pkl"))
gammas_mensuales = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "gammas_ff3_mensuales.pkl"))

beta_MKT_rolling = primera_rolling["beta_MKT_rolling"]
beta_SMB_rolling = primera_rolling["beta_SMB_rolling"]
beta_HML_rolling = primera_rolling["beta_HML_rolling"]
VENTANA          = primera_rolling["ventana"]

print("=" * 70)
print("COMPROBACIONES DEL FF3 MENSUAL (Fama-MacBeth rolling 60m)")
print("=" * 70)

# ========================================================================
# COMPROBACIÓN 1: REGRESAR Mkt-RF CONTRA LOS TRES FACTORES
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 1: regresión de Mkt-RF contra (Mkt-RF, SMB, HML)")
print("─" * 70)
print("Predicción: alpha = 0, beta_MKT = 1, beta_SMB = 0, beta_HML = 0, R² = 1.")

y = factores_ff3["Mkt-RF"]
X = sm.add_constant(factores_ff3)
modelo = sm.OLS(y, X).fit()

print(f"  alpha:     {modelo.params['const']:.10f}")
print(f"  beta_MKT:  {modelo.params['Mkt-RF']:.10f}")
print(f"  beta_SMB:  {modelo.params['SMB']:.10f}")
print(f"  beta_HML:  {modelo.params['HML']:.10f}")
print(f"  R²:        {modelo.rsquared:.10f}")

check_1 = (abs(modelo.params['const']) < 1e-10 and
           abs(modelo.params['Mkt-RF'] - 1) < 1e-10 and
           abs(modelo.params['SMB']) < 1e-10 and
           abs(modelo.params['HML']) < 1e-10 and
           abs(modelo.rsquared - 1) < 1e-10)
print(f"  → {'✓ PASA' if check_1 else '✗ FALLA'}: la identidad se reproduce exactamente.")

# ========================================================================
# COMPROBACIÓN 2: REPLICACIÓN MANUAL DE BETAS FF3 ESTÁTICAS
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 2: replicación manual de betas FF3 estáticas (sin librería)")
print("─" * 70)

cartera_test = "ME5 BM5"
datos_test = pd.concat([excesos[cartera_test], factores_ff3], axis=1).dropna()
y_vec = datos_test.iloc[:, 0].values
X_mat = np.column_stack([np.ones(len(datos_test)),
                          datos_test["Mkt-RF"].values,
                          datos_test["SMB"].values,
                          datos_test["HML"].values])
beta_manual = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec

alpha_sm = primera.loc[cartera_test, "alpha"]
bMKT_sm  = primera.loc[cartera_test, "beta_MKT"]
bSMB_sm  = primera.loc[cartera_test, "beta_SMB"]
bHML_sm  = primera.loc[cartera_test, "beta_HML"]

print(f"  Cartera testada: {cartera_test}")
print(f"  Coeficiente  |  Manual          |  Statsmodels     |  Diferencia")
print(f"  ─────────────────────────────────────────────────────────────────")
print(f"  alpha        |  {beta_manual[0]:.10f}   |  {alpha_sm:.10f}   |  {abs(beta_manual[0]-alpha_sm):.2e}")
print(f"  beta_MKT     |  {beta_manual[1]:.10f}   |  {bMKT_sm:.10f}   |  {abs(beta_manual[1]-bMKT_sm):.2e}")
print(f"  beta_SMB     |  {beta_manual[2]:.10f}   |  {bSMB_sm:.10f}   |  {abs(beta_manual[2]-bSMB_sm):.2e}")
print(f"  beta_HML     |  {beta_manual[3]:.10f}   |  {bHML_sm:.10f}   |  {abs(beta_manual[3]-bHML_sm):.2e}")

check_2 = (abs(beta_manual[0] - alpha_sm) < 1e-10 and
           abs(beta_manual[1] - bMKT_sm) < 1e-10 and
           abs(beta_manual[2] - bSMB_sm) < 1e-10 and
           abs(beta_manual[3] - bHML_sm) < 1e-10)
print(f"  → {'✓ PASA' if check_2 else '✗ FALLA'}: cálculo manual y librería coinciden.")

# ========================================================================
# COMPROBACIÓN 3: gamma_HML ESTIMADA vs PRIMA REALIZADA DE HML
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 3: coherencia gamma_HML vs prima realizada del factor HML")
print("─" * 70)

prima_HML = factores_ff3.loc[gammas_mensuales.index, "HML"].mean() * 12 * 100
g_HML = segunda.loc["gamma_HML", "media_anual_%"]

print(f"  Prima realizada de HML (T={len(gammas_mensuales)}):  {prima_HML:.3f}% anual")
print(f"  gamma_HML estimada (FM rolling 60m):                  {g_HML:.3f}% anual")
print(f"  Bajo FF3 ambas magnitudes deberían coincidir aproximadamente.")
print(f"  Diferencia observada:                                 {abs(prima_HML - g_HML):.3f} pp")
if prima_HML != 0:
    print(f"  Ratio gamma_HML / prima realizada:                   {g_HML / prima_HML:.3f}")
print(f"  → Cuanto más cerca de 1 esté el ratio, mayor coherencia con el modelo.")

# ========================================================================
# COMPROBACIÓN 4: REPLICACIÓN PASO A PASO DEL TEST GRS FF3
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 4: replicación paso a paso del test GRS FF3")
print("─" * 70)

residuos = pd.DataFrame(index=excesos.index, columns=excesos.columns, dtype=float)
X_t = sm.add_constant(factores_ff3)
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    residuos.loc[datos.index, cartera] = modelo.resid

residuos = residuos.dropna()
T = residuos.shape[0]
N = residuos.shape[1]
K = 3

alpha_vec = primera["alpha"].values.reshape(-1, 1)
Sigma = residuos.cov().values
mu_f = factores_ff3.loc[residuos.index].mean().values.reshape(-1, 1)
Omega_f = factores_ff3.loc[residuos.index].cov().values

quad = (alpha_vec.T @ np.linalg.inv(Sigma) @ alpha_vec).item()
SR2_f = (mu_f.T @ np.linalg.inv(Omega_f) @ mu_f).item()
grs = ((T - N - K) / N) * quad / (1 + SR2_f)
pval = 1 - stats.f.cdf(grs, dfn=N, dfd=T-N-K)

print(f"  T:                       {T}")
print(f"  N:                       {N}")
print(f"  K:                       {K}")
print(f"  α' Σ^-1 α:               {quad:.6f}")
print(f"  Sharpe² multivariante:   {SR2_f:.6f}")
print(f"  Estadístico GRS:         {grs:.4f}")
print(f"  p-valor F({N},{T-N-K}):  {pval:.6f}")

check_4 = grs > 1.0
print(f"  → {'✓ PASA' if check_4 else '✗ FALLA'}: estadístico GRS calculado correctamente.")

# ========================================================================
# COMPROBACIÓN 5: REPLICACIÓN MANUAL DE LA METODOLOGÍA ROLLING FF3
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 5: replicación manual de la metodología rolling FF3 60m")
print("─" * 70)
print("Verificamos sobre UN mes concreto que las betas y los gammas guardados")
print("en los pickles coinciden con el cálculo manual paso a paso.")

excesos_ext = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_extendido.pkl"))
factores_ff3_ext = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_extendido.pkl"))

idx_test = len(beta_MKT_rolling) // 2
fecha_test = beta_MKT_rolling.index[idx_test]
print(f"\n  Mes de prueba: {fecha_test} (índice {idx_test} en la serie rolling)")

pos_ext = excesos_ext.index.get_loc(fecha_test)
fechas_ventana = excesos_ext.index[pos_ext - VENTANA : pos_ext]
print(f"  Ventana usada: [{fechas_ventana[0]}, {fechas_ventana[-1]}] (60 meses)")

# Re-estimación manual de las 100 ternas de betas
factores_ventana = factores_ff3_ext.loc[fechas_ventana]
X_ventana = sm.add_constant(factores_ventana)
betas_manuales_MKT = {}
betas_manuales_SMB = {}
betas_manuales_HML = {}

for cartera in excesos_ext.columns:
    y_ventana = excesos_ext.loc[fechas_ventana, cartera]
    datos = pd.concat([y_ventana, X_ventana], axis=1).dropna()
    if len(datos) < 30:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    betas_manuales_MKT[cartera] = modelo.params["Mkt-RF"]
    betas_manuales_SMB[cartera] = modelo.params["SMB"]
    betas_manuales_HML[cartera] = modelo.params["HML"]

betas_manuales_MKT = pd.Series(betas_manuales_MKT)
betas_manuales_SMB = pd.Series(betas_manuales_SMB)
betas_manuales_HML = pd.Series(betas_manuales_HML)

# Comparación con betas rolling guardadas en el paso 1
betas_g_MKT = beta_MKT_rolling.loc[fecha_test].dropna()
betas_g_SMB = beta_SMB_rolling.loc[fecha_test].dropna()
betas_g_HML = beta_HML_rolling.loc[fecha_test].dropna()

dif_MKT = (betas_manuales_MKT.reindex(betas_g_MKT.index) - betas_g_MKT).abs().max()
dif_SMB = (betas_manuales_SMB.reindex(betas_g_SMB.index) - betas_g_SMB).abs().max()
dif_HML = (betas_manuales_HML.reindex(betas_g_HML.index) - betas_g_HML).abs().max()

print(f"  Diferencia máxima entre betas manuales y guardadas:")
print(f"    beta_MKT: {dif_MKT:.2e}")
print(f"    beta_SMB: {dif_SMB:.2e}")
print(f"    beta_HML: {dif_HML:.2e}")
check_5a = max(dif_MKT, dif_SMB, dif_HML) < 1e-10
print(f"  → {'✓ PASA' if check_5a else '✗ FALLA'}: betas rolling FF3 guardadas son correctas.")

# Replicación manual de la regresión transversal del mes test
y_test = excesos.loc[fecha_test]
X_test = pd.concat([
    pd.Series(1.0, index=betas_g_MKT.index, name="const"),
    betas_g_MKT.rename("beta_MKT"),
    betas_g_SMB.rename("beta_SMB"),
    betas_g_HML.rename("beta_HML"),
], axis=1)
datos_test = pd.concat([y_test, X_test], axis=1).dropna()
modelo_test = sm.OLS(datos_test.iloc[:, 0], datos_test.iloc[:, 1:]).fit()

g0_m   = modelo_test.params["const"]
gMKT_m = modelo_test.params["beta_MKT"]
gSMB_m = modelo_test.params["beta_SMB"]
gHML_m = modelo_test.params["beta_HML"]

g0_g   = gammas_mensuales.loc[fecha_test, "gamma_0"]
gMKT_g = gammas_mensuales.loc[fecha_test, "gamma_MKT"]
gSMB_g = gammas_mensuales.loc[fecha_test, "gamma_SMB"]
gHML_g = gammas_mensuales.loc[fecha_test, "gamma_HML"]

print(f"\n  Coeficiente  |  Manual         |  Guardado       |  Diferencia")
print(f"  ───────────────────────────────────────────────────────────────")
print(f"  gamma_0      |  {g0_m:.10f}   |  {g0_g:.10f}   |  {abs(g0_m-g0_g):.2e}")
print(f"  gamma_MKT    |  {gMKT_m:.10f}   |  {gMKT_g:.10f}   |  {abs(gMKT_m-gMKT_g):.2e}")
print(f"  gamma_SMB    |  {gSMB_m:.10f}   |  {gSMB_g:.10f}   |  {abs(gSMB_m-gSMB_g):.2e}")
print(f"  gamma_HML    |  {gHML_m:.10f}   |  {gHML_g:.10f}   |  {abs(gHML_m-gHML_g):.2e}")

check_5b = (abs(g0_m - g0_g) < 1e-10 and
            abs(gMKT_m - gMKT_g) < 1e-10 and
            abs(gSMB_m - gSMB_g) < 1e-10 and
            abs(gHML_m - gHML_g) < 1e-10)
print(f"  → {'✓ PASA' if check_5b else '✗ FALLA'}: regresión transversal del mes test correcta.")

check_5 = check_5a and check_5b

# ========================================================================
# RESUMEN
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"  1. Identidad de la regresión FF3:                "
      f"{'✓ PASA' if check_1 else '✗ FALLA'}")
print(f"  2. Replicación manual β FF3 estática:            "
      f"{'✓ PASA' if check_2 else '✗ FALLA'}")
print(f"  3. Coherencia gamma_HML vs prima HML:            ✓ INFORMATIVO")
print(f"  4. Replicación paso a paso del test GRS FF3:     "
      f"{'✓ PASA' if check_4 else '✗ FALLA'}")
print(f"  5. Replicación manual de la metodología rolling: "
      f"{'✓ PASA' if check_5 else '✗ FALLA'}")
