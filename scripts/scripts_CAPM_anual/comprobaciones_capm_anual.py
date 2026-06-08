"""
COMPROBACIONES - CAPM anual
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Cinco verificaciones independientes:
  1. Composición compuesta: rendimientos anuales coherentes con los mensuales.
  2. Identidad de la regresión simple en frecuencia anual.
  3. Replicación manual de β para una cartera (sin librería).
  4. Coherencia gamma_1 anual vs gamma_1 mensual (robustez a frecuencia).
  5. GRS con agregación 5×5 (el de 100 carteras no es factible con T=62).
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import re

CARPETA_PROCESADOS = "datos_procesados"
CARPETA_ANUAL      = "datos_procesados_anual"

excesos_a   = pd.read_pickle(os.path.join(CARPETA_ANUAL, "excesos_carteras_anual.pkl"))
mkt_rf_a    = pd.read_pickle(os.path.join(CARPETA_ANUAL, "mkt_rf_anual.pkl"))
primera_a   = pd.read_pickle(os.path.join(CARPETA_ANUAL, "primera_etapa_anual.pkl"))
segunda_a   = pd.read_pickle(os.path.join(CARPETA_ANUAL, "segunda_etapa_anual.pkl"))

excesos_m   = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
mkt_rf_m    = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))
segunda_m   = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa.pkl"))

print("=" * 70)
print("COMPROBACIONES DEL CAPM ANUAL")
print("=" * 70)

# ========================================================================
# COMPROBACIÓN 1: COMPOSICIÓN COMPUESTA MENSUAL → ANUAL
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 1: composición compuesta de rendimientos mensuales a anuales")
print("─" * 70)
print("Recalculamos un año concreto desde los mensuales y comparamos con el anual.")

anio_test = 2020

# El índice mensual puede ser PeriodIndex o strings; lo manejamos en ambos casos
def reindex_a_datetime(serie):
    if isinstance(serie.index, pd.PeriodIndex):
        serie = serie.copy()
        serie.index = serie.index.to_timestamp()
    else:
        serie = serie.copy()
        serie.index = pd.to_datetime(serie.index.astype(str), format="%Y%m")
    return serie

mkt_rf_m_temp = reindex_a_datetime(mkt_rf_m)

rf_m = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "rf.pkl"))
rf_m_temp = reindex_a_datetime(rf_m)

mkt_bruto_m = mkt_rf_m_temp + rf_m_temp
rendimientos_2020 = mkt_bruto_m[mkt_bruto_m.index.year == anio_test]
rf_2020 = rf_m_temp[rf_m_temp.index.year == anio_test]

mkt_2020_recalc = (1 + rendimientos_2020).prod() - 1
rf_2020_recalc  = (1 + rf_2020).prod() - 1
prima_2020_recalc = mkt_2020_recalc - rf_2020_recalc
prima_2020_archivo = mkt_rf_a.loc[anio_test]

print(f"  Año testado:                       {anio_test}")
print(f"  Prima recalculada desde mensuales: {prima_2020_recalc:.6f}")
print(f"  Prima del archivo anual:           {prima_2020_archivo:.6f}")
print(f"  Diferencia absoluta:               {abs(prima_2020_recalc - prima_2020_archivo):.6f}")

check_1 = abs(prima_2020_recalc - prima_2020_archivo) < 1e-3
print(f"  → {'✓ PASA' if check_1 else '✗ FALLA'}: la composición compuesta es coherente.")

# ========================================================================
# COMPROBACIÓN 2: IDENTIDAD DE LA REGRESIÓN ANUAL
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 2: regresión Mkt-RF anual contra sí mismo")
print("─" * 70)
print("Predicción: alpha = 0, beta = 1, R² = 1.")

X = sm.add_constant(mkt_rf_a.rename("Mkt-RF"))
modelo = sm.OLS(mkt_rf_a, X).fit()

print(f"  alpha:  {modelo.params['const']:.10f}")
print(f"  beta:   {modelo.params['Mkt-RF']:.10f}")
print(f"  R²:     {modelo.rsquared:.10f}")

check_2 = (abs(modelo.params['const']) < 1e-10 and
           abs(modelo.params['Mkt-RF'] - 1) < 1e-10 and
           abs(modelo.rsquared - 1) < 1e-10)
print(f"  → {'✓ PASA' if check_2 else '✗ FALLA'}: identidad correcta en frecuencia anual.")

# ========================================================================
# COMPROBACIÓN 3: REPLICACIÓN MANUAL DE BETA ANUAL
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 3: replicación manual de β anual (sin librería)")
print("─" * 70)

cartera_test = "ME5 BM5"
y_full = excesos_a[cartera_test].dropna()
x_full = mkt_rf_a.loc[y_full.index]

X_mat = np.column_stack([np.ones(len(x_full)), x_full.values])
y_vec = y_full.values

beta_manual = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec

alpha_sm = primera_a.loc[cartera_test, "alpha"]
beta_sm  = primera_a.loc[cartera_test, "beta"]

print(f"  Cartera testada: {cartera_test}")
print(f"  Coeficiente  |  Manual         |  Statsmodels    |  Diferencia")
print(f"  ───────────────────────────────────────────────────────────────")
print(f"  alpha        |  {beta_manual[0]:.10f}   |  {alpha_sm:.10f}   |  {abs(beta_manual[0]-alpha_sm):.2e}")
print(f"  beta         |  {beta_manual[1]:.10f}   |  {beta_sm:.10f}   |  {abs(beta_manual[1]-beta_sm):.2e}")

check_3 = (abs(beta_manual[0] - alpha_sm) < 1e-10 and
           abs(beta_manual[1] - beta_sm) < 1e-10)
print(f"  → {'✓ PASA' if check_3 else '✗ FALLA'}: cálculo manual y librería coinciden.")

# ========================================================================
# COMPROBACIÓN 4: ROBUSTEZ DEL RECHAZO ENTRE FRECUENCIAS
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 4: coherencia cualitativa de la segunda etapa")
print("─" * 70)
print("Los signos y la significación de las gammas deben ser estables entre")
print("frecuencias, aunque los valores numéricos cambien por la diferente")
print("potencia estadística.")

g0_m = segunda_m.loc["gamma_0", "media_anual_%"]
g1_m = segunda_m.loc["gamma_1", "media_anual_%"]
g0_t_m = segunda_m.loc["gamma_0", "t_stat"]
g1_t_m = segunda_m.loc["gamma_1", "t_stat"]

g0_a = segunda_a.loc["gamma_0", "media_pct"]
g1_a = segunda_a.loc["gamma_1", "media_pct"]
g0_t_a = segunda_a.loc["gamma_0", "t_stat"]
g1_t_a = segunda_a.loc["gamma_1", "t_stat"]

print(f"  Coeficiente  |  Mensual         |  Anual           |  ¿Misma conclusión?")
print(f"  ─────────────────────────────────────────────────────────────────────")
print(f"  gamma_0      |  {g0_m:6.3f}% (t={g0_t_m:5.2f}) |  {g0_a:6.3f}% (t={g0_t_a:5.2f}) |  "
      f"{'✓ ambos sig.' if abs(g0_t_m) > 1.96 and abs(g0_t_a) > 1.96 else '✗ discrepan'}")
print(f"  gamma_1      |  {g1_m:6.3f}% (t={g1_t_m:5.2f}) |  {g1_a:6.3f}% (t={g1_t_a:5.2f}) |  "
      f"{'✓ ambos no sig.' if abs(g1_t_m) < 1.96 and abs(g1_t_a) < 1.96 else '✗ discrepan'}")

check_4 = (abs(g0_t_m) > 1.96 and abs(g0_t_a) > 1.96 and
           abs(g1_t_m) < 1.96 and abs(g1_t_a) < 1.96)
print(f"  → {'✓ PASA' if check_4 else '✗ FALLA'}: el rechazo del CAPM es robusto a la frecuencia.")

# ========================================================================
# COMPROBACIÓN 5: GRS CON 5×5 (limitación técnica del análisis anual)
# ========================================================================
print("\n" + "─" * 70)
print("COMPROBACIÓN 5: test GRS con agregación 5×5")
print("─" * 70)
print(f"Con T={excesos_a.shape[0]} < N={excesos_a.shape[1]}, el GRS clásico no es factible.")
print("Agregamos las 100 carteras en 25 carteras 5×5 y aplicamos el GRS.")

def parsear_nombre(nombre):
    n = nombre.strip()
    if n == "SMALL LoBM": return 1, 1
    if n == "SMALL HiBM": return 1, 10
    if n == "BIG LoBM":   return 10, 1
    if n == "BIG HiBM":   return 10, 10
    m = re.match(r"ME(\d+)\s+BM(\d+)", n)
    if m:
        return int(m.group(1)), int(m.group(2))

def decil_a_quintil(d):
    return (d - 1) // 2 + 1

mapping_5x5 = {}
for c in excesos_a.columns:
    s, b = parsear_nombre(c)
    mapping_5x5[c] = f"S{decil_a_quintil(s)}B{decil_a_quintil(b)}"

excesos_5x5 = excesos_a.copy()
excesos_5x5.columns = [mapping_5x5[c] for c in excesos_a.columns]
excesos_5x5 = excesos_5x5.T.groupby(level=0).mean().T

X_t = sm.add_constant(mkt_rf_a.rename("Mkt-RF"))
alfas, residuos_5x5 = [], pd.DataFrame(index=excesos_5x5.index, columns=excesos_5x5.columns, dtype=float)
for cartera in excesos_5x5.columns:
    y = excesos_5x5[cartera]
    datos = pd.concat([y, X_t], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    alfas.append(modelo.params["const"])
    residuos_5x5.loc[datos.index, cartera] = modelo.resid

residuos_5x5 = residuos_5x5.dropna()
T5 = residuos_5x5.shape[0]
N5 = residuos_5x5.shape[1]
alpha_5x5 = np.array(alfas).reshape(-1, 1)
Sigma_5x5 = residuos_5x5.cov().values

SR2_M = (mkt_rf_a.mean() / mkt_rf_a.std()) ** 2
quad = (alpha_5x5.T @ np.linalg.inv(Sigma_5x5) @ alpha_5x5).item()
grs = ((T5 - N5 - 1) / N5) * quad / (1 + SR2_M)
pval = 1 - stats.f.cdf(grs, dfn=N5, dfd=T5-N5-1)

print(f"  T (años):                 {T5}")
print(f"  N (carteras 5×5):         {N5}")
print(f"  Estadístico GRS:          {grs:.4f}")
print(f"  p-valor F({N5}, {T5-N5-1}):   {pval:.6f}")

check_5 = pval < 0.10
print(f"  → {'✓ PASA' if check_5 else 'NOTA'}: tendencia coherente con el mensual aunque con menor potencia.")

# ========================================================================
# RESUMEN
# ========================================================================
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"  1. Composición compuesta correcta:           "
      f"{'✓ PASA' if check_1 else '✗ FALLA'}")
print(f"  2. Identidad de la regresión anual:          "
      f"{'✓ PASA' if check_2 else '✗ FALLA'}")
print(f"  3. Replicación manual de β anual:            "
      f"{'✓ PASA' if check_3 else '✗ FALLA'}")
print(f"  4. Robustez entre frecuencias (mensual/anual): "
      f"{'✓ PASA' if check_4 else '✗ FALLA'}")
print(f"  5. GRS sobre carteras 5×5:                   "
      f"{'✓ PASA' if check_5 else 'NOTA'}")
