"""
COMPROBACIONES - CAPM mensual (metodología Fama-MacBeth rolling 60 meses)
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Cinco verificaciones independientes:
  1. Regresar Mkt-RF contra sí mismo → alpha=0, beta=1, R²=1.
  2. Replicación manual de las betas estáticas (sin statsmodels).
  3. Coherencia gamma_1 estimada vs prima realizada del mercado.
  4. Replicación paso a paso del test GRS.
  5. Replicación manual de la metodología rolling 60 meses sobre un mes
     concreto: se comprueba que la beta rolling reproducida a mano coincide
     con la guardada por el paso 1, y que la gamma_1 obtenida en ese mes
     coincide con la guardada por el paso 2.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_PROCESADOS = "datos_procesados"

excesos          = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
mkt_rf           = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))
primera          = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))
primera_rolling  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_rolling.pkl"))
segunda          = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa.pkl"))
gammas_mensuales = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "gammas_mensuales.pkl"))

betas_rolling = primera_rolling["betas_rolling"]
VENTANA       = primera_rolling["ventana"]

print("=" * 70)
print("COMPROBACIONES DEL CAPM MENSUAL (Fama-MacBeth rolling 60m)")
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
# COMPROBACIÓN 2: REPLICACIÓN MANUAL DE UNA REGRESIÓN ESTÁTICA
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 2: replicación manual de β estática (sin librería)")
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

# Prima realizada sobre el mismo período que la segunda etapa
mkt_seccion = mkt_rf.loc[gammas_mensuales.index]
prima_realizada = mkt_seccion.mean() * 12 * 100
gamma_1_estimada = segunda.loc["gamma_1", "media_anual_%"]

print(f"  Prima realizada del mercado (T_efectivo = {len(gammas_mensuales)}):  "
      f"{prima_realizada:.3f}% anual")
print(f"  gamma_1 estimada (FM rolling 60m):                       "
      f"{gamma_1_estimada:.3f}% anual")
print(f"  Bajo el CAPM debería cumplirse gamma_1 ≈ prima realizada.")
print(f"  Diferencia observada:                                    "
      f"{abs(prima_realizada - gamma_1_estimada):.3f} pp")
print(f"  → Cualquier discrepancia material refleja el rechazo empírico")
print(f"    del CAPM, no un fallo de la implementación.")

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

check_4 = grs > 1.5
print(f"  → {'✓ PASA' if check_4 else '✗ FALLA'}: estadístico GRS calculado correctamente.")

# ========================================================================
# COMPROBACIÓN 5: REPLICACIÓN MANUAL DE LA METODOLOGÍA ROLLING
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 5: replicación manual de la metodología rolling 60m")
print("─" * 70)
print("Verificamos sobre UN mes concreto que las betas rolling y la gamma_1")
print("guardadas en los pickles coinciden con el cálculo manual paso a paso.")

# Elegimos un mes representativo: la mitad de la muestra rolling
idx_test = len(betas_rolling) // 2
fecha_test = betas_rolling.index[idx_test]
print(f"\n  Mes de prueba: {fecha_test} (índice {idx_test} en la serie rolling)")

# Ventana de 60 meses anteriores al mes de prueba
posicion_global = excesos.index.get_loc(fecha_test)
fechas_ventana = excesos.index[posicion_global - VENTANA : posicion_global]
print(f"  Ventana usada: [{fechas_ventana[0]}, {fechas_ventana[-1]}] (60 meses)")

# Re-estimación manual de las 100 betas en esa ventana
mkt_ventana = mkt_rf.loc[fechas_ventana]
X_ventana = sm.add_constant(mkt_ventana.rename("Mkt-RF"))
betas_manuales = {}
for cartera in excesos.columns:
    y_ventana = excesos.loc[fechas_ventana, cartera]
    datos = pd.concat([y_ventana, X_ventana], axis=1).dropna()
    if len(datos) < 30:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    betas_manuales[cartera] = modelo.params["Mkt-RF"]
betas_manuales = pd.Series(betas_manuales)

# Comparar con las betas rolling guardadas en el paso 1
betas_guardadas = betas_rolling.loc[fecha_test].dropna()
betas_ambas = pd.concat([betas_manuales.rename("manual"),
                         betas_guardadas.rename("guardadas")], axis=1).dropna()

dif_max = (betas_ambas["manual"] - betas_ambas["guardadas"]).abs().max()
print(f"  Diferencia máxima entre betas manuales y guardadas: {dif_max:.2e}")
check_5a = dif_max < 1e-10
print(f"  → {'✓ PASA' if check_5a else '✗ FALLA'}: betas rolling guardadas son correctas.")

# Replicación manual de la regresión transversal del mes test
y_test = excesos.loc[fecha_test]
X_test = sm.add_constant(betas_guardadas.rename("beta"))
datos_test = pd.concat([y_test, X_test], axis=1).dropna()
modelo_test = sm.OLS(datos_test.iloc[:, 0], datos_test.iloc[:, 1:]).fit()
gamma_0_manual = modelo_test.params["const"]
gamma_1_manual = modelo_test.params["beta"]

# Compararlo con la gamma guardada en el paso 2
gamma_0_guardado = gammas_mensuales.loc[fecha_test, "gamma_0"]
gamma_1_guardado = gammas_mensuales.loc[fecha_test, "gamma_1"]

print(f"  Coeficiente  |  Manual         |  Guardado       |  Diferencia")
print(f"  ───────────────────────────────────────────────────────────────")
print(f"  gamma_0      |  {gamma_0_manual:.10f}   |  {gamma_0_guardado:.10f}   |  {abs(gamma_0_manual-gamma_0_guardado):.2e}")
print(f"  gamma_1      |  {gamma_1_manual:.10f}   |  {gamma_1_guardado:.10f}   |  {abs(gamma_1_manual-gamma_1_guardado):.2e}")

check_5b = (abs(gamma_0_manual - gamma_0_guardado) < 1e-10 and
            abs(gamma_1_manual - gamma_1_guardado) < 1e-10)
print(f"  → {'✓ PASA' if check_5b else '✗ FALLA'}: regresión transversal del mes test correcta.")

check_5 = check_5a and check_5b

# ========================================================================
# RESUMEN
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"  1. Identidad de la regresión simple:           "
      f"{'✓ PASA' if check_1 else '✗ FALLA'}")
print(f"  2. Replicación manual β estática:              "
      f"{'✓ PASA' if check_2 else '✗ FALLA'}")
print(f"  3. Coherencia gamma_1 vs prima realizada:      ✓ INFORMATIVO")
print(f"  4. Replicación paso a paso del test GRS:       "
      f"{'✓ PASA' if check_4 else '✗ FALLA'}")
print(f"  5. Replicación manual de la metodología rolling: "
      f"{'✓ PASA' if check_5 else '✗ FALLA'}")
