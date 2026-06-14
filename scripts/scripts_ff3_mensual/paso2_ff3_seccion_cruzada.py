"""
PASO 2 FF3 - Segunda etapa Fama-MacBeth: regresión de sección cruzada
TFG Valoración de Activos - Contraste empírico del FF3
Autor: Miguel Suárez Crespo

Segunda etapa en DOS versiones, coherentes con la primera etapa del paso 1
(estática y móvil), según indicación del tutor:

  (A) VERSIÓN ESTÁTICA (1 sola regresión transversal):
      bar{r_j} - bar{r_f} = gamma_0 + gamma_M beta_j_MKT
                          + gamma_SMB beta_j_SMB + gamma_HML beta_j_HML + u_j
      donde bar{r_j} es el exceso medio temporal de la cartera j sobre todo
      el periodo muestral, y las tres betas son las estáticas estimadas con
      las 754 observaciones (paso 1, parte A).
      Una sola regresión MCO con N=100 observaciones; errores estándar
      habituales. Análogo al ejercicio de Marín y Rubio (cuadros 11.7, 11.8).

  (B) VERSIÓN CON VENTANA MÓVIL (754 regresiones transversales):
      Para cada mes t:
        r_{j,t} - r_{f,t} = gamma_{0,t} + gamma_{M,t} beta_{j,M,t-1}
                            + gamma_{SMB,t} beta_{j,SMB,t-1}
                            + gamma_{HML,t} beta_{j,HML,t-1} + u_{j,t}
      donde las tres betas se han estimado con los 60 meses anteriores.
      Inferencia Fama-MacBeth: media temporal de los gammas y t-estadístico
      basado en la desviación típica de la serie.

Predicciones del FF3 (con excesos de rendimiento):
  gamma_0   = 0
  gamma_M   = E[r_M - r_f] (positivo, cercano a la prima realizada)
  gamma_SMB = E[SMB] (cercano a la prima realizada)
  gamma_HML = E[HML] (cercano a la prima realizada)
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

CARPETA_PROCESADOS = "datos_procesados"
CARPETA_RESULTADOS = "resultados/tablas/ff3_mensual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

# Carga
excesos       = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras_ff3.pkl"))
factores_ff3  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))
primera       = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3.pkl"))
primera_rolling = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3_rolling.pkl"))

beta_MKT_rolling = primera_rolling["beta_MKT_rolling"]
beta_SMB_rolling = primera_rolling["beta_SMB_rolling"]
beta_HML_rolling = primera_rolling["beta_HML_rolling"]
VENTANA          = primera_rolling["ventana"]

# Betas estáticas (paso 1A)
betas_estaticas = primera[["beta_MKT", "beta_SMB", "beta_HML"]]

print(f"Datos cargados (FF3):")
print(f"  - Excesos de rendimiento:  {excesos.shape[0]} meses x {excesos.shape[1]} carteras")
print(f"  - Betas estáticas:         {betas_estaticas.shape[0]} carteras x 3 factores")
print(f"  - Betas móviles:           {beta_MKT_rolling.shape[0]} meses x {beta_MKT_rolling.shape[1]} carteras (cada una)")

# ========================================================================
# (A) SEGUNDA ETAPA ESTÁTICA FF3 - 1 SOLA REGRESIÓN TRANSVERSAL
# ========================================================================
print("\n" + "=" * 70)
print("(A) SEGUNDA ETAPA ESTÁTICA FF3 - 1 regresión transversal (N=100)")
print("=" * 70)

y_estatica = excesos.mean()  # rendimiento medio temporal de cada cartera
X_estatica = sm.add_constant(betas_estaticas)

datos_est = pd.concat([y_estatica.rename("y"), X_estatica], axis=1).dropna()
modelo_estatico = sm.OLS(datos_est["y"], datos_est[["const", "beta_MKT", "beta_SMB", "beta_HML"]]).fit()
modelo_estatico_hc1 = sm.OLS(datos_est["y"], datos_est[["const", "beta_MKT", "beta_SMB", "beta_HML"]]).fit(cov_type="HC1")

print(f"\nN (carteras):        {int(modelo_estatico.nobs)}")
print(f"R^2:                 {modelo_estatico.rsquared:.4f}")
print(f"R^2 ajustado:        {modelo_estatico.rsquared_adj:.4f}")

print(f"\n{'Coef':<12} {'Estim.':>10} {'ee_MCO':>10} {'t_MCO':>8} {'p_MCO':>8} {'ee_HC1':>10} {'t_HC1':>8}")
for k in ["const", "beta_MKT", "beta_SMB", "beta_HML"]:
    print(f"{k:<12} {modelo_estatico.params[k]:>10.6f} "
          f"{modelo_estatico.bse[k]:>10.6f} {modelo_estatico.tvalues[k]:>8.3f} "
          f"{modelo_estatico.pvalues[k]:>8.4f} "
          f"{modelo_estatico_hc1.bse[k]:>10.6f} {modelo_estatico_hc1.tvalues[k]:>8.3f}")

print(f"\nCoeficientes anualizados (%):")
for k in ["const", "beta_MKT", "beta_SMB", "beta_HML"]:
    print(f"  {k:<12} {modelo_estatico.params[k]*12*100:>8.3f}%")

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
print(f"(B) SEGUNDA ETAPA CON VENTANA MÓVIL FF3 - {len(beta_MKT_rolling)} regresiones")
print("=" * 70)
print(f"Para cada mes t, regresión transversal con las tres betas estimadas")
print(f"con los {VENTANA} meses anteriores.")

gammas_list = []
fechas_t = []

for fecha_t in beta_MKT_rolling.index:
    y       = excesos.loc[fecha_t]
    beta_M  = beta_MKT_rolling.loc[fecha_t]
    beta_S  = beta_SMB_rolling.loc[fecha_t]
    beta_H  = beta_HML_rolling.loc[fecha_t]

    X = pd.concat([
        pd.Series(1.0, index=beta_M.index, name="const"),
        beta_M.rename("beta_MKT"),
        beta_S.rename("beta_SMB"),
        beta_H.rename("beta_HML"),
    ], axis=1)
    datos = pd.concat([y, X], axis=1).dropna()
    if len(datos) < 10:
        continue
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    gammas_list.append(modelo.params.values)
    fechas_t.append(fecha_t)

gammas = pd.DataFrame(gammas_list,
                     columns=["gamma_0", "gamma_MKT", "gamma_SMB", "gamma_HML"],
                     index=fechas_t)

T = len(gammas)

resumen_rolling = pd.DataFrame(index=["gamma_0", "gamma_MKT", "gamma_SMB", "gamma_HML"])
resumen_rolling["media"]          = gammas.mean().values
resumen_rolling["desv_tipica"]    = gammas.std().values
resumen_rolling["error_estandar"] = resumen_rolling["desv_tipica"] / np.sqrt(T)
resumen_rolling["t_stat"]         = resumen_rolling["media"] / resumen_rolling["error_estandar"]
resumen_rolling["p_valor"]        = 2 * (1 - stats.t.cdf(resumen_rolling["t_stat"].abs(), df=T-1))
resumen_rolling["media_anual_%"]  = resumen_rolling["media"] * 12 * 100

# Primas realizadas
primas_realizadas_anual = {
    "gamma_0":   np.nan,
    "gamma_MKT": factores_ff3.loc[gammas.index, "Mkt-RF"].mean() * 12 * 100,
    "gamma_SMB": factores_ff3.loc[gammas.index, "SMB"].mean() * 12 * 100,
    "gamma_HML": factores_ff3.loc[gammas.index, "HML"].mean() * 12 * 100,
}
resumen_rolling["prima_realizada_anual_%"] = [primas_realizadas_anual[k] for k in resumen_rolling.index]

print(f"\nNúmero de regresiones transversales: {T}")
print(resumen_rolling.round(5))

# ========================================================================
# COMPARACIÓN DE LAS DOS VERSIONES
# ========================================================================
print("\n" + "=" * 70)
print("COMPARACIÓN: VERSIÓN ESTÁTICA vs VERSIÓN MÓVIL FF3")
print("=" * 70)
print(f"{'Coeficiente':<13} {'Estática (% anual, t)':>26} {'Móvil (% anual, t)':>25} {'Prima realiz.':>15}")
print(f"{'-'*80}")
for k in ["const", "beta_MKT", "beta_SMB", "beta_HML"]:
    k_rolling = "gamma_0" if k == "const" else "gamma_" + k.split("_")[1]
    e_coef = modelo_estatico.params[k] * 12 * 100
    e_t    = modelo_estatico.tvalues[k]
    m_coef = resumen_rolling.loc[k_rolling, "media_anual_%"]
    m_t    = resumen_rolling.loc[k_rolling, "t_stat"]
    prima  = primas_realizadas_anual[k_rolling] if k != "const" else None
    prima_str = f"{prima:>10.3f}%" if prima is not None else "—"
    print(f"{k:<13} {e_coef:>15.3f} (t={e_t:>5.2f})  {m_coef:>14.3f} (t={m_t:>5.2f})   {prima_str:>15}")

print(f"\nBajo el FF3 se espera:")
print(f"  gamma_0   ≈ 0")
print(f"  gamma_k > 0 y cercano a la prima realizada del factor k, para k = MKT, SMB, HML")

# ========================================================================
# GUARDADO
# ========================================================================
resumen_estatica.to_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3_estatica.pkl"))
resumen_estatica.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa_ff3_estatica.csv"))

gammas.to_pickle(os.path.join(CARPETA_PROCESADOS, "gammas_ff3_mensuales.pkl"))
resumen_rolling.to_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3.pkl"))
resumen_rolling.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa_ff3.csv"))

print(f"\n✓ Versión estática guardada en '{CARPETA_PROCESADOS}/segunda_etapa_ff3_estatica.pkl'")
print(f"✓ Versión móvil guardada en '{CARPETA_PROCESADOS}/segunda_etapa_ff3.pkl'")
print(f"✓ Tablas CSV exportadas a '{CARPETA_RESULTADOS}/'")
print("✓ Listos para ejecutar paso3_ff3_contrastes.py")
