"""
PASO 5 ANUAL - Comprobaciones de los resultados CAPM (datos anuales)
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Verifica reproducibilidad cruzando los resultados guardados con cálculos
manuales independientes.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")

excesos     = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_anual.pkl"))
mkt_rf      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf_anual.pkl"))
primera     = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_anual.pkl"))
gammas      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "gammas_anuales.pkl"))
segunda_rolling = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_anual.pkl"))
segunda_est = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_anual_estatica.pkl"))

print("=" * 70)
print("COMPROBACIONES (DATOS ANUALES)")
print("=" * 70)

# 1) Primera etapa: recálculo manual de una cartera
cart = excesos.columns[0]
y = excesos[cart]
X = sm.add_constant(mkt_rf)
datos = pd.concat([y, X], axis=1).dropna()
m = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
diff_alpha = abs(m.params["const"] - primera.loc[cart, "alpha"])
diff_beta  = abs(m.params["Mkt-RF"] - primera.loc[cart, "beta"])
print(f"\n1) Primera etapa, cartera {cart}:")
print(f"   alpha manual: {m.params['const']:.6f}   alpha guardado: {primera.loc[cart, 'alpha']:.6f}   diff: {diff_alpha:.2e}")
print(f"   beta manual:  {m.params['Mkt-RF']:.6f}   beta guardado:  {primera.loc[cart, 'beta']:.6f}   diff: {diff_beta:.2e}")

# 2) Segunda etapa estática
y_est = excesos.mean()
beta_est = primera["beta"]
datos_est = pd.concat([y_est.rename("y"), sm.add_constant(beta_est.rename("beta"))], axis=1).dropna()
m_est = sm.OLS(datos_est["y"], datos_est[["const", "beta"]]).fit()
diff_g0 = abs(m_est.params["const"] - segunda_est.loc["const", "coef"])
diff_g1 = abs(m_est.params["beta"]  - segunda_est.loc["beta", "coef"])
print(f"\n2) Segunda etapa estática:")
print(f"   gamma_0 manual: {m_est.params['const']:.6f}   guardado: {segunda_est.loc['const', 'coef']:.6f}   diff: {diff_g0:.2e}")
print(f"   gamma_1 manual: {m_est.params['beta']:.6f}   guardado: {segunda_est.loc['beta', 'coef']:.6f}   diff: {diff_g1:.2e}")

# 3) Segunda etapa móvil: media manual de las gammas
T = len(gammas)
diff_g0_m = abs(gammas["gamma_0"].mean() - segunda_rolling.loc["gamma_0", "media"])
diff_g1_m = abs(gammas["gamma_1"].mean() - segunda_rolling.loc["gamma_1", "media"])
print(f"\n3) Segunda etapa móvil (T={T}):")
print(f"   media gamma_0 manual: {gammas['gamma_0'].mean():.6f}   guardado: {segunda_rolling.loc['gamma_0', 'media']:.6f}   diff: {diff_g0_m:.2e}")
print(f"   media gamma_1 manual: {gammas['gamma_1'].mean():.6f}   guardado: {segunda_rolling.loc['gamma_1', 'media']:.6f}   diff: {diff_g1_m:.2e}")

# 4) t-stat móvil
t0_manual = gammas["gamma_0"].mean() / (gammas["gamma_0"].std() / np.sqrt(T))
t1_manual = gammas["gamma_1"].mean() / (gammas["gamma_1"].std() / np.sqrt(T))
print(f"\n4) t-estadísticos móvil:")
print(f"   t(gamma_0) manual: {t0_manual:.3f}   guardado: {segunda_rolling.loc['gamma_0', 't_stat']:.3f}")
print(f"   t(gamma_1) manual: {t1_manual:.3f}   guardado: {segunda_rolling.loc['gamma_1', 't_stat']:.3f}")

# 5) Prima del mercado realizada
prima_anual = mkt_rf.mean() * 100
print(f"\n5) Prima realizada del mercado: {prima_anual:.3f}% anual")

print("\n" + "=" * 70)
print("Si las 'diff' son ≈ 0 (1e-10 o menos), el código es reproducible.")
print("=" * 70)
