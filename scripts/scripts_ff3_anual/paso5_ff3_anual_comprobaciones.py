"""
PASO 5 ANUAL FF3 - Comprobaciones de reproducibilidad (anual)
Autor: Miguel Suárez Crespo
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")

excesos      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_anual.pkl"))
factores_ff3 = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_anual.pkl"))
primera      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3_anual.pkl"))
gammas       = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "gammas_ff3_anuales.pkl"))
segunda_r    = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3_anual.pkl"))
segunda_e    = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3_anual_estatica.pkl"))

print("=" * 70)
print("COMPROBACIONES FF3 ANUAL")
print("=" * 70)

# 1) Primera etapa multifactorial
cart = excesos.columns[0]
y = excesos[cart]
datos = pd.concat([y, factores_ff3], axis=1).dropna()
X = sm.add_constant(datos[["Mkt-RF", "SMB", "HML"]])
m = sm.OLS(datos.iloc[:, 0], X).fit()
print(f"\n1) Primera etapa, cartera {cart}:")
for nombre, var in [("beta_MKT", "Mkt-RF"), ("beta_SMB", "SMB"), ("beta_HML", "HML"), ("alpha", "const")]:
    diff = abs(m.params[var] - primera.loc[cart, nombre])
    print(f"   {nombre}: manual={m.params[var]:.6f}  guardado={primera.loc[cart, nombre]:.6f}  diff={diff:.2e}")

# 2) Segunda etapa estática
y_est = excesos.mean()
X_est = sm.add_constant(primera[["beta_MKT", "beta_SMB", "beta_HML"]])
datos_est = pd.concat([y_est.rename("y"), X_est], axis=1).dropna()
m_est = sm.OLS(datos_est["y"], datos_est[["const", "beta_MKT", "beta_SMB", "beta_HML"]]).fit()
print(f"\n2) Segunda etapa estática:")
for k in ["const", "beta_MKT", "beta_SMB", "beta_HML"]:
    diff = abs(m_est.params[k] - segunda_e.loc[k, "coef"])
    print(f"   {k}: manual={m_est.params[k]:.6f}  guardado={segunda_e.loc[k, 'coef']:.6f}  diff={diff:.2e}")

# 3) Segunda etapa móvil
T = len(gammas)
print(f"\n3) Segunda etapa móvil (T={T}):")
for k in ["gamma_0", "gamma_MKT", "gamma_SMB", "gamma_HML"]:
    media_manual = gammas[k].mean()
    media_guardada = segunda_r.loc[k, "media"]
    diff = abs(media_manual - media_guardada)
    t_manual = media_manual / (gammas[k].std() / np.sqrt(T))
    t_guardado = segunda_r.loc[k, "t_stat"]
    print(f"   {k}: media manual={media_manual:.6f}   guardado={media_guardada:.6f}   diff={diff:.2e}   t={t_manual:.3f}")

print("\n" + "=" * 70)
print("Diferencias deberían ser ≈ 0 (1e-10 o menos).")
print("=" * 70)
