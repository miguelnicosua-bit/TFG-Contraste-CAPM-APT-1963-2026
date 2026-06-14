"""
PASO 3 ANUAL FF3 - Contrastes formales (frecuencia anual)
Autor: Miguel Suárez Crespo

1. GRS (limitación T < N + K + 1 documentada).
2. White (heterocedasticidad sobre los residuos de la 1ª etapa estática).
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smdiag
from scipy import stats

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")
CARPETA_RESULTADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "resultados", "tablas", "ff3_anual")
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_anual.pkl"))
factores_ff3 = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_anual.pkl"))
primera      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3_anual.pkl"))

print(f"T = {excesos.shape[0]}, N = {excesos.shape[1]}, K = 3")

# Recálculo de residuos
residuos = pd.DataFrame(index=excesos.index, columns=excesos.columns, dtype=float)
X_t = sm.add_constant(factores_ff3)
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    residuos.loc[datos.index, cartera] = modelo.resid

residuos_grs = residuos.dropna()
T_grs = residuos_grs.shape[0]
N_grs = residuos_grs.shape[1]
K = 3

# ---- TEST GRS (con limitación) ----
print("\n" + "=" * 70)
print("CONTRASTE 1: TEST GRS FF3 (LIMITACIÓN POR T < N + K)")
print("=" * 70)
print(f"T={T_grs}   N={N_grs}   K={K}   T-N-K={T_grs-N_grs-K}")

resumen_grs = pd.DataFrame()

if T_grs > N_grs + K:
    alpha = primera["alpha"].values.reshape(-1, 1)
    Sigma = residuos_grs.cov().values
    factores_grs = factores_ff3.loc[residuos_grs.index]
    mu_f = factores_grs.mean().values.reshape(-1, 1)
    Omega_f = factores_grs.cov().values
    SR2 = (mu_f.T @ np.linalg.inv(Omega_f) @ mu_f).item()
    Sigma_inv = np.linalg.inv(Sigma)
    alpha_quad = (alpha.T @ Sigma_inv @ alpha).item()
    grs_stat = ((T_grs - N_grs - K) / N_grs) * alpha_quad / (1 + SR2)
    grs_pvalue = 1 - stats.f.cdf(grs_stat, dfn=N_grs, dfd=T_grs - N_grs - K)
    print(f"\nF = {grs_stat:.4f}   p = {grs_pvalue:.6f}")
    resumen_grs = pd.DataFrame({
        "estadistico": [grs_stat], "p_valor": [grs_pvalue],
        "T": [T_grs], "N": [N_grs], "K": [K],
    })
else:
    print(f"\n→ Test GRS NO APLICABLE: T={T_grs} < N+K={N_grs+K}")
    print(f"  No se puede invertir la matriz Sigma. Resultado coherente con")
    print(f"  el análisis mensual (T=754 >> N=100), que sí permite el contraste.")
    resumen_grs = pd.DataFrame({"observacion": ["no aplicable: T < N+K"]})

# ---- TEST DE WHITE ----
print("\n" + "=" * 70)
print("CONTRASTE 2: WHITE SOBRE LA 1ª ETAPA ESTÁTICA FF3")
print("=" * 70)
X_t_w = sm.add_constant(factores_ff3)
resultados_w = []
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t_w], axis=1).dropna()
    Xc = datos[["const", "Mkt-RF", "SMB", "HML"]]
    yc = datos[cartera]
    modelo = sm.OLS(yc, Xc).fit()
    try:
        w_lm, w_lm_p, _, _ = smdiag.het_white(modelo.resid, Xc)
    except Exception:
        w_lm, w_lm_p = np.nan, np.nan
    resultados_w.append({
        "cartera": cartera,
        "White_LM": w_lm,
        "White_p":  w_lm_p,
        "White_rechaza": int(w_lm_p < 0.05) if not np.isnan(w_lm_p) else 0,
    })
hetero_df = pd.DataFrame(resultados_w).set_index("cartera")
n_w_5 = hetero_df["White_rechaza"].sum()
n_w_1 = (hetero_df["White_p"] < 0.01).sum()
print(f"\nRechazan H0 al 5%:  {n_w_5}/100")
print(f"Rechazan H0 al 1%:  {n_w_1}/100")
print(f"p medio: {hetero_df['White_p'].mean():.4f}   p mediano: {hetero_df['White_p'].median():.4f}")

# ---- RESUMEN ----
print("\n" + "=" * 70)
print("RESUMEN CONTRASTES FF3 ANUAL")
print("=" * 70)
print(f"GRS:                         no aplicable (T < N + K)")
print(f"Heterocedasticidad White:    {n_w_5}/100 rechazan al 5%")

# GUARDADO
hetero_df.to_csv(os.path.join(CARPETA_RESULTADOS, "test_heterocedasticidad_ff3_anual.csv"))
resumen_grs.to_csv(os.path.join(CARPETA_RESULTADOS, "test_grs_ff3_anual.csv"), index=False)
print(f"\n✓ Tablas exportadas a {CARPETA_RESULTADOS}/")
print("✓ Listo para paso4_ff3_anual_graficos.py")
