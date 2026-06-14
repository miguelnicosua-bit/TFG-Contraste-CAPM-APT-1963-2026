"""
PASO 2 - Segunda etapa Fama-MacBeth: regresión de sección cruzada
TFG Valoración de Activos - Contraste empírico del CAPM
Autor: Miguel Suárez Crespo

Implementación de la segunda etapa en DOS versiones, coherentes con la
primera etapa del paso 1 (estática y móvil), según indicación del tutor
A. Rodríguez Sampayo:

  (A) VERSIÓN ESTÁTICA (1 sola regresión transversal):
        bar{r_j} - bar{r_f} = gamma_0 + gamma_1 * beta_j_hat + u_j
      donde bar{r_j} es el exceso medio temporal de la cartera j sobre
      todo el periodo muestral, y beta_j_hat es la beta estática estimada
      con las 754 observaciones (paso 1, parte A).
      Una regresión MCO con N=100 observaciones; errores estándar habituales.
      Análoga al ejercicio de los cuadros 11.7 y 11.8 de Marín y Rubio
      pero con N=100 carteras en lugar de 25.

  (B) VERSIÓN CON VENTANA MÓVIL (754 regresiones transversales):
      Para cada mes t del periodo muestral:
        r_{j,t} - r_{f,t} = gamma_{0,t} + gamma_{1,t} * beta_{j,M,t-1} + u_{j,t}
      donde beta_{j,M,t-1} se ha estimado con los 60 meses anteriores [t-60, t-1].
      Inferencia Fama-MacBeth: media temporal de los gammas, t-estadístico
      basado en la desviación típica de la serie.

Predicciones del CAPM (con excesos de rendimiento):
  - gamma_0 = 0
  - gamma_1 > 0 y aproximadamente igual a E[r_M - r_f]
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# ========================================================================
# CARGA DE DATOS
# ========================================================================
CARPETA_PROCESADOS = "datos_procesados"
CARPETA_RESULTADOS = "resultados/tablas/capm_mensual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
mkt_rf  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))

# Betas estáticas (paso 1A) y betas móviles (paso 1B)
primera         = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))
primera_rolling = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_rolling.pkl"))
betas_estaticas = primera["beta"]
betas_rolling   = primera_rolling["betas_rolling"]
VENTANA         = primera_rolling["ventana"]

print(f"Datos cargados:")
print(f"  - Excesos de rendimiento:    {excesos.shape[0]} meses x {excesos.shape[1]} carteras")
print(f"  - Betas estáticas:           {len(betas_estaticas)} carteras (1 beta por cartera)")
print(f"  - Betas móviles:             {betas_rolling.shape[0]} meses x {betas_rolling.shape[1]} carteras")
print(f"  - Ventana móvil:             {VENTANA} meses")

# ========================================================================
# (A) SEGUNDA ETAPA ESTÁTICA - 1 SOLA REGRESIÓN TRANSVERSAL
# ========================================================================
# Análogo al ejercicio de los cuadros 11.7 y 11.8 de Marín y Rubio.
# Sobre la sección cruzada de las 100 carteras:
#   - variable dependiente: media temporal del exceso de rendimiento
#   - regresor: beta estática estimada en el paso 1A
# Errores estándar habituales de MCO (heterocedasticidad-consistentes HC1
# también reportados como referencia).

print("\n" + "=" * 70)
print("(A) SEGUNDA ETAPA ESTÁTICA - 1 regresión transversal (N=100)")
print("=" * 70)

# Rendimiento medio temporal de cada cartera (excluyendo NaN cartera a cartera)
y_estatica = excesos.mean()

# Diseño: constante + beta estática
X_estatica = sm.add_constant(betas_estaticas.rename("beta"))

# Alineamos por si alguna cartera tuviese problemas
datos_est = pd.concat([y_estatica.rename("y"), X_estatica], axis=1).dropna()
modelo_estatico = sm.OLS(datos_est["y"], datos_est[["const", "beta"]]).fit()

print(f"\nN (carteras):        {int(modelo_estatico.nobs)}")
print(f"R^2:                 {modelo_estatico.rsquared:.4f}")
print(f"R^2 ajustado:        {modelo_estatico.rsquared_adj:.4f}")
print(f"\n              Coef        ee (MCO)      t-stat     p-valor")
print(f"  gamma_0     {modelo_estatico.params['const']:.6f}   "
      f"{modelo_estatico.bse['const']:.6f}    "
      f"{modelo_estatico.tvalues['const']:.3f}    "
      f"{modelo_estatico.pvalues['const']:.4f}")
print(f"  gamma_1     {modelo_estatico.params['beta']:.6f}   "
      f"{modelo_estatico.bse['beta']:.6f}    "
      f"{modelo_estatico.tvalues['beta']:.3f}    "
      f"{modelo_estatico.pvalues['beta']:.4f}")
print(f"\n  gamma_0 anualizado (%):  {modelo_estatico.params['const']*12*100:.3f}%")
print(f"  gamma_1 anualizado (%):  {modelo_estatico.params['beta']*12*100:.3f}%")

# Errores estándar robustos a heterocedasticidad (HC1, equivalente al
# White con corrección de grados de libertad), de referencia
modelo_estatico_hc1 = sm.OLS(datos_est["y"], datos_est[["const", "beta"]]).fit(cov_type="HC1")
print(f"\n  [Referencia: errores estándar robustos HC1]")
print(f"  gamma_0: ee_HC1 = {modelo_estatico_hc1.bse['const']:.6f}   "
      f"t_HC1 = {modelo_estatico_hc1.tvalues['const']:.3f}")
print(f"  gamma_1: ee_HC1 = {modelo_estatico_hc1.bse['beta']:.6f}    "
      f"t_HC1 = {modelo_estatico_hc1.tvalues['beta']:.3f}")

resumen_estatica = pd.DataFrame({
    "coef":            modelo_estatico.params,
    "ee_MCO":          modelo_estatico.bse,
    "t_MCO":           modelo_estatico.tvalues,
    "p_MCO":           modelo_estatico.pvalues,
    "ee_HC1":          modelo_estatico_hc1.bse,
    "t_HC1":           modelo_estatico_hc1.tvalues,
    "p_HC1":           modelo_estatico_hc1.pvalues,
    "coef_anual_%":    modelo_estatico.params * 12 * 100,
})

# ========================================================================
# (B) SEGUNDA ETAPA CON VENTANA MÓVIL - 754 REGRESIONES TRANSVERSALES
# ========================================================================
print("\n" + "=" * 70)
print(f"(B) SEGUNDA ETAPA CON VENTANA MÓVIL - {len(betas_rolling)} regresiones transversales")
print("=" * 70)
print(f"Metodología Fama-MacBeth (1973): para cada mes t, la regresión")
print(f"transversal usa las betas estimadas con los {VENTANA} meses anteriores.")

gammas_0 = []
gammas_1 = []
fechas_t = []

for fecha_t in betas_rolling.index:
    y = excesos.loc[fecha_t]
    beta_t = betas_rolling.loc[fecha_t]
    X = sm.add_constant(beta_t.rename("beta"))
    datos = pd.concat([y, X], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_0.append(modelo.params["const"])
    gammas_1.append(modelo.params["beta"])
    fechas_t.append(fecha_t)

gammas = pd.DataFrame({"gamma_0": gammas_0, "gamma_1": gammas_1}, index=fechas_t)
T = len(gammas)

resumen_rolling = pd.DataFrame(index=["gamma_0", "gamma_1"])
resumen_rolling["media"]          = gammas.mean().values
resumen_rolling["desv_tipica"]    = gammas.std().values
resumen_rolling["error_estandar"] = resumen_rolling["desv_tipica"] / np.sqrt(T)
resumen_rolling["t_stat"]         = resumen_rolling["media"] / resumen_rolling["error_estandar"]
resumen_rolling["p_valor"]        = 2 * (1 - stats.t.cdf(resumen_rolling["t_stat"].abs(), df=T-1))
resumen_rolling["media_anual_%"]  = resumen_rolling["media"] * 12 * 100

print(f"\nNúmero de regresiones transversales: {T}")
print(resumen_rolling.round(5))

# ========================================================================
# COMPARACIÓN DE LAS DOS VERSIONES
# ========================================================================
prima_mkt_anual = mkt_rf.mean() * 12 * 100

print("\n" + "=" * 70)
print("COMPARACIÓN: VERSIÓN ESTÁTICA vs VERSIÓN MÓVIL")
print("=" * 70)
print(f"{'':20} {'gamma_0 (% anual)':>22} {'gamma_1 (% anual)':>22}")
print(f"{'-'*64}")
print(f"{'Versión estática':20} "
      f"{modelo_estatico.params['const']*12*100:>17.3f} (t={modelo_estatico.tvalues['const']:>5.2f})"
      f"  {modelo_estatico.params['beta']*12*100:>14.3f} (t={modelo_estatico.tvalues['beta']:>5.2f})")
print(f"{'Versión móvil':20} "
      f"{resumen_rolling.loc['gamma_0','media_anual_%']:>17.3f} (t={resumen_rolling.loc['gamma_0','t_stat']:>5.2f})"
      f"  {resumen_rolling.loc['gamma_1','media_anual_%']:>14.3f} (t={resumen_rolling.loc['gamma_1','t_stat']:>5.2f})")
print(f"\nPrima realizada del mercado E[r_M - r_f]: {prima_mkt_anual:.3f}% anual")
print(f"\nBajo el CAPM se espera:")
print(f"  gamma_0 ≈ 0")
print(f"  gamma_1 > 0 y cercano a la prima realizada del mercado")

# ========================================================================
# GUARDADO
# ========================================================================
# Versión estática
resumen_estatica.to_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_estatica.pkl"))
resumen_estatica.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa_estatica.csv"))

# Versión móvil (mismos nombres de archivo que antes, para compatibilidad)
gammas.to_pickle(os.path.join(CARPETA_PROCESADOS, "gammas_mensuales.pkl"))
resumen_rolling.to_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa.pkl"))
resumen_rolling.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa.csv"))

print(f"\n✓ Segunda etapa estática guardada en '{CARPETA_PROCESADOS}/segunda_etapa_estatica.pkl'")
print(f"✓ Segunda etapa móvil guardada en '{CARPETA_PROCESADOS}/segunda_etapa.pkl'")
print(f"✓ Tablas CSV exportadas a '{CARPETA_RESULTADOS}/'")
print("✓ Listos para ejecutar paso3_contrastes.py")
