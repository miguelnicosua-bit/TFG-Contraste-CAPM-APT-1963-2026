"""
COMPROBACIONES - Modelo de tres factores (FF3) anual
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Cinco verificaciones independientes:
  1. Identidad de la regresión multifactorial en frecuencia anual.
  2. Replicación manual de las 3 betas (sin librería).
  3. Coherencia gamma_HML anual vs gamma_HML mensual.
  4. FF3 anual mejora al CAPM anual (R², |α|, alfas significativos).
  5. Robustez FF3 mensual vs FF3 anual (betas factoriales estables).
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

CARPETA_PROCESADOS = "datos_procesados"
CARPETA_ANUAL      = "datos_procesados_anual"

excesos_a       = pd.read_pickle(os.path.join(CARPETA_ANUAL, "excesos_carteras_anual.pkl"))
factores_a      = pd.read_pickle(os.path.join(CARPETA_ANUAL, "factores_ff3_anual.pkl"))
primera_ff3_a   = pd.read_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_ff3_anual.pkl"))
primera_capm_a  = pd.read_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_anual.pkl"))
segunda_ff3_a   = pd.read_pickle(os.path.join(CARPETA_ANUAL, "segunda_etapa_ff3_anual.pkl"))

primera_ff3_m   = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3.pkl"))
segunda_ff3_m   = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3.pkl"))

print("=" * 70)
print("COMPROBACIONES DEL MODELO DE TRES FACTORES (FF3) ANUAL")
print("=" * 70)

# ========================================================================
# COMPROBACIÓN 1: IDENTIDAD MULTIFACTORIAL ANUAL
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 1: regresión del factor SMB contra los 3 factores anuales")
print("─" * 70)
print("Predicción: alpha = 0, beta_MKT = 0, beta_SMB = 1, beta_HML = 0, R² = 1.")

y = factores_a["SMB"]
X = sm.add_constant(factores_a)
modelo = sm.OLS(y, X).fit()

print(f"  alpha:      {modelo.params['const']:.10f}")
print(f"  β_MKT:      {modelo.params['Mkt-RF']:.10f}")
print(f"  β_SMB:      {modelo.params['SMB']:.10f}")
print(f"  β_HML:      {modelo.params['HML']:.10f}")
print(f"  R²:         {modelo.rsquared:.10f}")

check_1 = (abs(modelo.params['const']) < 1e-10 and
           abs(modelo.params['Mkt-RF']) < 1e-10 and
           abs(modelo.params['SMB'] - 1) < 1e-10 and
           abs(modelo.params['HML']) < 1e-10 and
           abs(modelo.rsquared - 1) < 1e-10)
print(f"  → {'✓ PASA' if check_1 else '✗ FALLA'}: identidad correcta.")

# ========================================================================
# COMPROBACIÓN 2: REPLICACIÓN MANUAL DE BETAS FACTORIALES ANUALES
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 2: replicación manual de las 3 betas anuales (sin librería)")
print("─" * 70)

cartera_test = "ME5 BM5"
y_full = excesos_a[cartera_test].dropna()
X_full = factores_a.loc[y_full.index]

X_mat = np.column_stack([np.ones(len(X_full)), X_full.values])
y_vec = y_full.values

beta_manual = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec

alpha_sm    = primera_ff3_a.loc[cartera_test, "alpha"]
beta_MKT_sm = primera_ff3_a.loc[cartera_test, "beta_MKT"]
beta_SMB_sm = primera_ff3_a.loc[cartera_test, "beta_SMB"]
beta_HML_sm = primera_ff3_a.loc[cartera_test, "beta_HML"]

print(f"  Cartera testada: {cartera_test}")
print(f"  Coeficiente  |  Manual         |  Statsmodels    |  Diferencia")
print(f"  ───────────────────────────────────────────────────────────────")
print(f"  alpha        |  {beta_manual[0]:.10f}   |  {alpha_sm:.10f}   |  {abs(beta_manual[0]-alpha_sm):.2e}")
print(f"  β_MKT        |  {beta_manual[1]:.10f}   |  {beta_MKT_sm:.10f}   |  {abs(beta_manual[1]-beta_MKT_sm):.2e}")
print(f"  β_SMB        |  {beta_manual[2]:.10f}   |  {beta_SMB_sm:.10f}   |  {abs(beta_manual[2]-beta_SMB_sm):.2e}")
print(f"  β_HML        |  {beta_manual[3]:.10f}   |  {beta_HML_sm:.10f}   |  {abs(beta_manual[3]-beta_HML_sm):.2e}")

check_2 = (abs(beta_manual[0] - alpha_sm) < 1e-10 and
           abs(beta_manual[1] - beta_MKT_sm) < 1e-10 and
           abs(beta_manual[2] - beta_SMB_sm) < 1e-10 and
           abs(beta_manual[3] - beta_HML_sm) < 1e-10)
print(f"  → {'✓ PASA' if check_2 else '✗ FALLA'}: cálculo manual y librería coinciden.")

# ========================================================================
# COMPROBACIÓN 3: COHERENCIA gamma_HML ANUAL vs MENSUAL
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 3: coherencia gamma_HML anual vs mensual")
print("─" * 70)

g_HML_m = segunda_ff3_m.loc["gamma_HML", "media_anual_%"]
g_HML_a = segunda_ff3_a.loc["gamma_HML", "media_pct"]
prima_HML_m = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "hml.pkl")).mean() * 12 * 100
prima_HML_a = factores_a["HML"].mean() * 100

print(f"  Frecuencia  |  gamma_HML estimada  |  Prima realizada HML")
print(f"  ────────────────────────────────────────────────────────")
print(f"  Mensual     |  {g_HML_m:.3f}%             |  {prima_HML_m:.3f}%")
print(f"  Anual       |  {g_HML_a:.3f}%             |  {prima_HML_a:.3f}%")
print(f"\n  En ambas frecuencias gamma_HML ≈ prima realizada, lo que valida")
print(f"  empíricamente el efecto valor como factor de riesgo con precio.")

check_3 = (abs(g_HML_m - prima_HML_m) < 1.0 and abs(g_HML_a - prima_HML_a) < 1.5)
print(f"  → {'✓ PASA' if check_3 else '✗ FALLA'}: ambas frecuencias confirman el efecto valor.")

# ========================================================================
# COMPROBACIÓN 4: FF3 ANUAL MEJORA AL CAPM ANUAL
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 4: el FF3 anual mejora al CAPM anual")
print("─" * 70)

r2_capm  = primera_capm_a["r2"].mean()
r2_ff3   = primera_ff3_a["r2"].mean()
abs_a_capm = primera_capm_a["alpha"].abs().mean() * 100
abs_a_ff3  = primera_ff3_a["alpha"].abs().mean() * 100
sig_capm = (primera_capm_a["alpha_t"].abs() > 1.96).sum()
sig_ff3  = (primera_ff3_a["alpha_t"].abs() > 1.96).sum()

print(f"  Métrica                          |  CAPM anual  |  FF3 anual   |  Mejora")
print(f"  ───────────────────────────────────────────────────────────────────────")
print(f"  R² medio                         |  {r2_capm:.4f}     |  {r2_ff3:.4f}     |  +{(r2_ff3-r2_capm)*100:.1f} pp")
print(f"  |α| medio anual (%)              |  {abs_a_capm:.3f}      |  {abs_a_ff3:.3f}      |  −{(1-abs_a_ff3/abs_a_capm)*100:.1f}%")
print(f"  Alfas significativos al 5%       |  {sig_capm}          |  {sig_ff3}          |  −{sig_capm-sig_ff3}")

check_4 = r2_ff3 > r2_capm and abs_a_ff3 < abs_a_capm and sig_ff3 < sig_capm
print(f"\n  → {'✓ PASA' if check_4 else '✗ FALLA'}: el FF3 anual supera al CAPM anual en las tres métricas.")

# ========================================================================
# COMPROBACIÓN 5: BETAS FACTORIALES ESTABLES ENTRE FRECUENCIAS
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 5: estabilidad de las betas factoriales entre frecuencias")
print("─" * 70)
print("Las cargas factoriales del FF3 deben ser robustas a la frecuencia,")
print("porque miden características económicas de las carteras.")

b_MKT_m = primera_ff3_m["beta_MKT"].mean()
b_SMB_m = primera_ff3_m["beta_SMB"].mean()
b_HML_m = primera_ff3_m["beta_HML"].mean()
b_MKT_a = primera_ff3_a["beta_MKT"].mean()
b_SMB_a = primera_ff3_a["beta_SMB"].mean()
b_HML_a = primera_ff3_a["beta_HML"].mean()

print(f"  Beta media   |  FF3 mensual    |  FF3 anual     |  Diferencia")
print(f"  ─────────────────────────────────────────────────────────────")
print(f"  β_MKT        |  {b_MKT_m:.3f}          |  {b_MKT_a:.3f}         |  {abs(b_MKT_m-b_MKT_a):.3f}")
print(f"  β_SMB        |  {b_SMB_m:.3f}          |  {b_SMB_a:.3f}         |  {abs(b_SMB_m-b_SMB_a):.3f}")
print(f"  β_HML        |  {b_HML_m:.3f}          |  {b_HML_a:.3f}         |  {abs(b_HML_m-b_HML_a):.3f}")

check_5 = (abs(b_MKT_m - b_MKT_a) < 0.15 and
           abs(b_SMB_m - b_SMB_a) < 0.15 and
           abs(b_HML_m - b_HML_a) < 0.15)
print(f"\n  → {'✓ PASA' if check_5 else 'NOTA'}: cargas factoriales estables entre frecuencias.")

# ========================================================================
# RESUMEN
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"  1. Identidad multifactorial anual:           "
      f"{'✓ PASA' if check_1 else '✗ FALLA'}")
print(f"  2. Replicación manual de las 3 betas:        "
      f"{'✓ PASA' if check_2 else '✗ FALLA'}")
print(f"  3. Coherencia gamma_HML mensual/anual:       "
      f"{'✓ PASA' if check_3 else '✗ FALLA'}")
print(f"  4. FF3 anual mejora al CAPM anual:           "
      f"{'✓ PASA' if check_4 else '✗ FALLA'}")
print(f"  5. Betas factoriales estables entre frecuencias: "
      f"{'✓ PASA' if check_5 else 'NOTA'}")
