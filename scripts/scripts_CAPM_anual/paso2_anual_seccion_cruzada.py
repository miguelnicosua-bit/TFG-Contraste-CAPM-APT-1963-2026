"""
PASO 2 ANUAL - Segunda etapa Fama-MacBeth (frecuencia anual)
TFG Valoración de Activos - Contraste empírico del CAPM
Autor: Miguel Suárez Crespo

Replica el paso 2 mensual en las dos versiones:

  (A) ESTÁTICA: 1 regresión transversal sobre los rendimientos medios
      temporales de las 100 carteras (N=100). Errores estándar MCO clásicos.
      Análoga a los cuadros 11.7 y 11.8 de Marín y Rubio (2011).

  (B) VENTANA MÓVIL: 62 regresiones transversales (una por año del periodo
      muestral) usando las betas rolling. Inferencia Fama-MacBeth.

Bajo el CAPM se espera gamma_0 = 0 y gamma_1 > 0, cercano a la prima
realizada del mercado anual.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CARPETA_PROCESADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "datos_procesados")
CARPETA_RESULTADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "resultados", "tablas", "capm_anual")
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_anual.pkl"))
mkt_rf  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf_anual.pkl"))

primera         = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_anual.pkl"))
primera_rolling = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_anual_rolling.pkl"))
betas_estaticas = primera["beta"]
betas_rolling   = primera_rolling["betas_rolling"]
VENTANA         = primera_rolling["ventana"]

print(f"Datos cargados:")
print(f"  Excesos anuales:     {excesos.shape[0]} años x {excesos.shape[1]} carteras")
print(f"  Betas estáticas:     {len(betas_estaticas)} carteras")
print(f"  Betas rolling:       {betas_rolling.shape[0]} años x {betas_rolling.shape[1]} carteras")
print(f"  Ventana móvil:       {VENTANA} años")

# ========================================================================
# (A) SEGUNDA ETAPA ESTÁTICA
# ========================================================================
print("\n" + "=" * 70)
print(f"(A) SEGUNDA ETAPA ESTÁTICA - 1 regresión transversal (N=100)")
print("=" * 70)

y_estatica = excesos.mean()
X_estatica = sm.add_constant(betas_estaticas.rename("beta"))
datos_est = pd.concat([y_estatica.rename("y"), X_estatica], axis=1).dropna()
modelo_estatico = sm.OLS(datos_est["y"], datos_est[["const", "beta"]]).fit()
modelo_estatico_hc1 = sm.OLS(datos_est["y"], datos_est[["const", "beta"]]).fit(cov_type="HC1")

print(f"\nN (carteras):  {int(modelo_estatico.nobs)}")
print(f"R^2:           {modelo_estatico.rsquared:.4f}")
print(f"\n            Coef         ee_MCO       t_MCO       p_MCO")
print(f"gamma_0   {modelo_estatico.params['const']:.6f}  "
      f"{modelo_estatico.bse['const']:.6f}  "
      f"{modelo_estatico.tvalues['const']:.3f}  "
      f"{modelo_estatico.pvalues['const']:.4f}")
print(f"gamma_1   {modelo_estatico.params['beta']:.6f}   "
      f"{modelo_estatico.bse['beta']:.6f}   "
      f"{modelo_estatico.tvalues['beta']:.3f}   "
      f"{modelo_estatico.pvalues['beta']:.4f}")
print(f"\ngamma_0 anual (%):  {modelo_estatico.params['const']*100:.3f}%")
print(f"gamma_1 anual (%):  {modelo_estatico.params['beta']*100:.3f}%")

resumen_estatica = pd.DataFrame({
    "coef":          modelo_estatico.params,
    "ee_MCO":        modelo_estatico.bse,
    "t_MCO":         modelo_estatico.tvalues,
    "p_MCO":         modelo_estatico.pvalues,
    "ee_HC1":        modelo_estatico_hc1.bse,
    "t_HC1":         modelo_estatico_hc1.tvalues,
    "p_HC1":         modelo_estatico_hc1.pvalues,
    "coef_anual_%":  modelo_estatico.params * 100,
})

# ========================================================================
# (B) SEGUNDA ETAPA CON VENTANA MÓVIL - 62 REGRESIONES TRANSVERSALES
# ========================================================================
print("\n" + "=" * 70)
print(f"(B) SEGUNDA ETAPA CON VENTANA MÓVIL - {len(betas_rolling)} regresiones")
print("=" * 70)

gammas_0, gammas_1, anios_t = [], [], []

for anio_t in betas_rolling.index:
    y = excesos.loc[anio_t]
    beta_t = betas_rolling.loc[anio_t]
    X = sm.add_constant(beta_t.rename("beta"))
    datos = pd.concat([y, X], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_0.append(modelo.params["const"])
    gammas_1.append(modelo.params["beta"])
    anios_t.append(anio_t)

gammas = pd.DataFrame({"gamma_0": gammas_0, "gamma_1": gammas_1}, index=anios_t)
T = len(gammas)

resumen_rolling = pd.DataFrame(index=["gamma_0", "gamma_1"])
resumen_rolling["media"]          = gammas.mean().values
resumen_rolling["desv_tipica"]    = gammas.std().values
resumen_rolling["error_estandar"] = resumen_rolling["desv_tipica"] / np.sqrt(T)
resumen_rolling["t_stat"]         = resumen_rolling["media"] / resumen_rolling["error_estandar"]
resumen_rolling["p_valor"]        = 2 * (1 - stats.t.cdf(resumen_rolling["t_stat"].abs(), df=T-1))
resumen_rolling["media_anual_%"]  = resumen_rolling["media"] * 100

print(f"\nNúmero de regresiones transversales: {T}")
print(resumen_rolling.round(5))

# ========================================================================
# COMPARACIÓN
# ========================================================================
prima_mkt_anual = mkt_rf.mean() * 100

print("\n" + "=" * 70)
print("COMPARACIÓN: ESTÁTICA vs MÓVIL (datos anuales)")
print("=" * 70)
print(f"{'':20} {'gamma_0 (% anual)':>22} {'gamma_1 (% anual)':>22}")
print(f"{'-'*64}")
print(f"{'Estática':20} "
      f"{modelo_estatico.params['const']*100:>15.3f} (t={modelo_estatico.tvalues['const']:>5.2f})"
      f"  {modelo_estatico.params['beta']*100:>12.3f} (t={modelo_estatico.tvalues['beta']:>5.2f})")
print(f"{'Móvil':20} "
      f"{resumen_rolling.loc['gamma_0','media_anual_%']:>15.3f} (t={resumen_rolling.loc['gamma_0','t_stat']:>5.2f})"
      f"  {resumen_rolling.loc['gamma_1','media_anual_%']:>12.3f} (t={resumen_rolling.loc['gamma_1','t_stat']:>5.2f})")
print(f"\nPrima realizada del mercado: {prima_mkt_anual:.3f}% anual")

# ========================================================================
# GUARDADO
# ========================================================================
resumen_estatica.to_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_anual_estatica.pkl"))
resumen_estatica.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa_anual_estatica.csv"))
gammas.to_pickle(os.path.join(CARPETA_PROCESADOS, "gammas_anuales.pkl"))
resumen_rolling.to_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_anual.pkl"))
resumen_rolling.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa_anual.csv"))

print(f"\n✓ Tablas exportadas a '{CARPETA_RESULTADOS}/'")
print("✓ Listos para ejecutar paso3_anual_contrastes.py")
