"""
PASO 2 ANUAL FF3 - Segunda etapa Fama-MacBeth (frecuencia anual)
Autor: Miguel Suárez Crespo
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")
CARPETA_RESULTADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "resultados", "tablas", "ff3_anual")
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3_anual.pkl"))
factores_ff3 = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3_anual.pkl"))
primera         = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3_anual.pkl"))
primera_rolling = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3_anual_rolling.pkl"))

beta_MKT_r = primera_rolling["beta_MKT_rolling"]
beta_SMB_r = primera_rolling["beta_SMB_rolling"]
beta_HML_r = primera_rolling["beta_HML_rolling"]
VENTANA    = primera_rolling["ventana"]

betas_estaticas = primera[["beta_MKT", "beta_SMB", "beta_HML"]]

# (A) ESTÁTICA - 1 regresión con N=100
print("=" * 70)
print("(A) SEGUNDA ETAPA ESTÁTICA FF3 ANUAL - 1 regresión (N=100)")
print("=" * 70)
y_est = excesos.mean()
X_est = sm.add_constant(betas_estaticas)
datos_est = pd.concat([y_est.rename("y"), X_est], axis=1).dropna()
modelo_est = sm.OLS(datos_est["y"], datos_est[["const", "beta_MKT", "beta_SMB", "beta_HML"]]).fit()
modelo_est_hc1 = sm.OLS(datos_est["y"], datos_est[["const", "beta_MKT", "beta_SMB", "beta_HML"]]).fit(cov_type="HC1")

print(f"\nR^2: {modelo_est.rsquared:.4f}")
print(f"\n{'':12} {'Coef':>10} {'ee':>10} {'t':>8} {'p':>8}")
for k in ["const", "beta_MKT", "beta_SMB", "beta_HML"]:
    print(f"{k:<12} {modelo_est.params[k]:>10.6f} {modelo_est.bse[k]:>10.6f} "
          f"{modelo_est.tvalues[k]:>8.3f} {modelo_est.pvalues[k]:>8.4f}")
print(f"\nCoeficientes anuales (%):")
for k in ["const", "beta_MKT", "beta_SMB", "beta_HML"]:
    print(f"  {k:<12} {modelo_est.params[k]*100:>8.3f}%")

resumen_estatica = pd.DataFrame({
    "coef":          modelo_est.params,
    "ee_MCO":        modelo_est.bse,
    "t_MCO":         modelo_est.tvalues,
    "p_MCO":         modelo_est.pvalues,
    "ee_HC1":        modelo_est_hc1.bse,
    "t_HC1":         modelo_est_hc1.tvalues,
    "p_HC1":         modelo_est_hc1.pvalues,
    "coef_anual_%":  modelo_est.params * 100,
})

# (B) ROLLING - 62 regresiones transversales
print("\n" + "=" * 70)
print(f"(B) SEGUNDA ETAPA ROLLING FF3 - {len(beta_MKT_r)} regresiones")
print("=" * 70)

gammas_list = []
anios_t = []
for anio_t in beta_MKT_r.index:
    y = excesos.loc[anio_t]
    X = pd.concat([
        pd.Series(1.0, index=excesos.columns, name="const"),
        beta_MKT_r.loc[anio_t].rename("beta_MKT"),
        beta_SMB_r.loc[anio_t].rename("beta_SMB"),
        beta_HML_r.loc[anio_t].rename("beta_HML"),
    ], axis=1)
    datos = pd.concat([y, X], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_list.append(modelo.params.values)
    anios_t.append(anio_t)

gammas = pd.DataFrame(gammas_list, columns=["gamma_0", "gamma_MKT", "gamma_SMB", "gamma_HML"], index=anios_t)
T = len(gammas)

resumen_rolling = pd.DataFrame(index=["gamma_0", "gamma_MKT", "gamma_SMB", "gamma_HML"])
resumen_rolling["media"]          = gammas.mean().values
resumen_rolling["desv_tipica"]    = gammas.std().values
resumen_rolling["error_estandar"] = resumen_rolling["desv_tipica"] / np.sqrt(T)
resumen_rolling["t_stat"]         = resumen_rolling["media"] / resumen_rolling["error_estandar"]
resumen_rolling["p_valor"]        = 2 * (1 - stats.t.cdf(resumen_rolling["t_stat"].abs(), df=T-1))
resumen_rolling["media_anual_%"]  = resumen_rolling["media"] * 100

primas_realiz = {
    "gamma_0":   np.nan,
    "gamma_MKT": factores_ff3.loc[gammas.index, "Mkt-RF"].mean() * 100,
    "gamma_SMB": factores_ff3.loc[gammas.index, "SMB"].mean() * 100,
    "gamma_HML": factores_ff3.loc[gammas.index, "HML"].mean() * 100,
}
resumen_rolling["prima_realiz_anual_%"] = [primas_realiz[k] for k in resumen_rolling.index]

print(f"\nT={T}")
print(resumen_rolling.round(5))

# COMPARACIÓN
print("\n" + "=" * 70)
print("COMPARACIÓN ESTÁTICA vs MÓVIL FF3 (anual)")
print("=" * 70)
print(f"{'Coef':<13} {'Estática (%, t)':>22} {'Móvil (%, t)':>22} {'Prima realiz.':>15}")
print("-" * 75)
for k in ["const", "beta_MKT", "beta_SMB", "beta_HML"]:
    k_r = "gamma_0" if k == "const" else "gamma_" + k.split("_")[1]
    e_c = modelo_est.params[k] * 100
    e_t = modelo_est.tvalues[k]
    m_c = resumen_rolling.loc[k_r, "media_anual_%"]
    m_t = resumen_rolling.loc[k_r, "t_stat"]
    prima = primas_realiz[k_r] if k != "const" else None
    prima_s = f"{prima:>10.2f}%" if prima is not None else "—"
    print(f"{k:<13} {e_c:>13.3f} (t={e_t:>5.2f})  {m_c:>13.3f} (t={m_t:>5.2f})   {prima_s:>15}")

# GUARDADO
resumen_estatica.to_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3_anual_estatica.pkl"))
resumen_estatica.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa_ff3_anual_estatica.csv"))
gammas.to_pickle(os.path.join(CARPETA_PROCESADOS, "gammas_ff3_anuales.pkl"))
resumen_rolling.to_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3_anual.pkl"))
resumen_rolling.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa_ff3_anual.csv"))
print("\n✓ Listo para paso3_ff3_anual_contrastes.py")
