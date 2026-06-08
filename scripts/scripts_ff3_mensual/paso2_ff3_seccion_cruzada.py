"""
PASO 2 FF3 - Segunda etapa Fama-MacBeth con tres factores
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Para cada mes t, regresamos los excesos de rendimiento de las 100 carteras
sobre las tres betas estimadas en el Paso 1:

    r_{j,t} - r_{f,t} = gamma_{0,t}
                      + gamma_{1,t} * beta_j^MKT
                      + gamma_{2,t} * beta_j^SMB
                      + gamma_{3,t} * beta_j^HML
                      + u_{j,t}

Luego promediamos los T pares de coeficientes y aplicamos el contraste
de Fama-MacBeth.

Predicciones del modelo FF3:
  - gamma_0 = 0
  - gamma_1 = E[r_M - r_f]   (prima de mercado realizada)
  - gamma_2 = E[SMB]         (prima del factor tamaño)
  - gamma_3 = E[HML]         (prima del factor valor)
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
CARPETA_RESULTADOS = "resultados/tablas/ff3_mensual"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

excesos       = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "excesos_carteras.pkl"))
factores_ff3  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))
primera_ff3   = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3.pkl"))

# Las tres betas estimadas en la primera etapa (una por cartera)
betas = primera_ff3[["beta_MKT", "beta_SMB", "beta_HML"]]

print(f"Datos cargados:")
print(f"  - Excesos de rendimiento: {excesos.shape}")
print(f"  - Betas factoriales:      {betas.shape}")

# ========================================================================
# REGRESIONES DE SECCIÓN CRUZADA MES A MES (CON 3 BETAS)
# ========================================================================
# Diseño fijo: constante + 3 betas estimadas.
X_design = sm.add_constant(betas)

gammas = []
fechas_t = []

for fecha in excesos.index:
    y = excesos.loc[fecha]
    datos = pd.concat([y, X_design], axis=1).dropna()
    if len(datos) < 10:
        continue
    y_clean = datos.iloc[:, 0]
    X_clean = datos.iloc[:, 1:]

    modelo = sm.OLS(y_clean, X_clean).fit()
    gammas.append(modelo.params.values)
    fechas_t.append(fecha)

gammas = pd.DataFrame(
    gammas,
    columns=["gamma_0", "gamma_MKT", "gamma_SMB", "gamma_HML"],
    index=fechas_t
)

print(f"\nNúmero de regresiones transversales: {len(gammas)}")

# ========================================================================
# AGREGACIÓN TEMPORAL: MEDIA Y T-ESTADÍSTICOS DE FAMA-MACBETH
# ========================================================================
T = len(gammas)

resumen = pd.DataFrame(index=gammas.columns)
resumen["media"]            = gammas.mean().values
resumen["desv_tipica"]      = gammas.std().values
resumen["error_estandar"]   = resumen["desv_tipica"] / np.sqrt(T)
resumen["t_stat"]           = resumen["media"] / resumen["error_estandar"]
resumen["p_valor"]          = pd.Series(
    2 * (1 - stats.t.cdf(resumen["t_stat"].abs(), df=T-1)),
    index=resumen.index
)
resumen["media_anual_%"]    = resumen["media"] * 12 * 100

print("\n" + "=" * 70)
print("RESULTADOS DE LA SEGUNDA ETAPA FAMA-MACBETH (FF3)")
print("=" * 70)
print(resumen.round(5))

# ========================================================================
# COMPARACIÓN CON LAS PRIMAS REALIZADAS DE CADA FACTOR
# ========================================================================
print("\n" + "=" * 70)
print("CONTRASTE: gamma_k estimada vs prima realizada de cada factor")
print("=" * 70)

primas_realizadas = factores_ff3.mean()  # Media de cada factor

comparacion = pd.DataFrame({
    "Factor":                ["MKT", "SMB", "HML"],
    "gamma estimada (% an)": [
        resumen.loc["gamma_MKT", "media_anual_%"],
        resumen.loc["gamma_SMB", "media_anual_%"],
        resumen.loc["gamma_HML", "media_anual_%"],
    ],
    "Prima realizada (% an)": [
        primas_realizadas["Mkt-RF"] * 12 * 100,
        primas_realizadas["SMB"] * 12 * 100,
        primas_realizadas["HML"] * 12 * 100,
    ],
    "t-stat de gamma": [
        resumen.loc["gamma_MKT", "t_stat"],
        resumen.loc["gamma_SMB", "t_stat"],
        resumen.loc["gamma_HML", "t_stat"],
    ],
    "p-valor": [
        resumen.loc["gamma_MKT", "p_valor"],
        resumen.loc["gamma_SMB", "p_valor"],
        resumen.loc["gamma_HML", "p_valor"],
    ],
})
print(comparacion.round(4).to_string(index=False))

print("\nBajo el modelo FF3, cada gamma_k debe coincidir con la prima")
print("realizada del factor correspondiente.")

# ========================================================================
# TEST GRS PARA EL MODELO FF3
# ========================================================================
print("\n" + "=" * 70)
print("TEST GRS PARA EL MODELO DE TRES FACTORES")
print("=" * 70)

# Necesitamos los residuos de la primera etapa FF3
residuos = pd.DataFrame(index=excesos.index, columns=excesos.columns, dtype=float)
X_t = sm.add_constant(factores_ff3)
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_t], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    residuos.loc[datos.index, cartera] = modelo.resid

# Eliminamos meses con NaN en cualquier cartera
residuos_grs = residuos.dropna()
T_grs = residuos_grs.shape[0]
N_grs = residuos_grs.shape[1]
K = 3  # número de factores

alpha_vec = primera_ff3["alpha"].values.reshape(-1, 1)
Sigma = residuos_grs.cov().values

# Matriz de covarianzas de los factores en el mismo período
factores_grs = factores_ff3.loc[residuos_grs.index]
mu_F = factores_grs.mean().values
Sigma_F = factores_grs.cov().values

# Forma cuadrática del estadístico GRS multifactorial
Sigma_inv = np.linalg.inv(Sigma)
alpha_quad = (alpha_vec.T @ Sigma_inv @ alpha_vec).item()

# Sharpe-squared multivariante de los factores
SR2_F = (mu_F @ np.linalg.inv(Sigma_F) @ mu_F)

grs_stat = ((T_grs - N_grs - K) / N_grs) * alpha_quad / (1 + SR2_F)
grs_pvalue = 1 - stats.f.cdf(grs_stat, dfn=N_grs, dfd=T_grs - N_grs - K)

print(f"T (meses):                       {T_grs}")
print(f"N (carteras):                    {N_grs}")
print(f"K (factores):                    {K}")
print(f"alpha' Sigma^-1 alpha:           {alpha_quad:.6f}")
print(f"Sharpe^2 multifactor:            {SR2_F:.6f}")
print(f"Estadístico GRS:                 {grs_stat:.4f}")
print(f"Distribución bajo H0:            F({N_grs}, {T_grs - N_grs - K})")
print(f"p-valor:                         {grs_pvalue:.6f}")
if grs_pvalue < 0.05:
    print(f"\n→ Se RECHAZA H0 al 5%. Los alfas son CONJUNTAMENTE distintos de cero")
    print(f"  incluso bajo el modelo FF3. El modelo de 3 factores mejora pero")
    print(f"  no captura completamente la sección cruzada de rendimientos.")
else:
    print(f"\n→ No se rechaza H0. Los alfas son CONJUNTAMENTE compatibles con cero")
    print(f"  bajo el modelo FF3. El modelo capta adecuadamente la sección cruzada.")

# ========================================================================
# COMPARACIÓN GRS: CAPM vs FF3
# ========================================================================
print("\n" + "=" * 70)
print("COMPARACIÓN DEL TEST GRS: CAPM vs FF3")
print("=" * 70)

# Recargamos el GRS del CAPM
capm = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))
mkt_rf_solo = factores_ff3["Mkt-RF"]

residuos_capm = pd.DataFrame(index=excesos.index, columns=excesos.columns, dtype=float)
X_capm = sm.add_constant(mkt_rf_solo)
for cartera in excesos.columns:
    y = excesos[cartera]
    datos = pd.concat([y, X_capm], axis=1).dropna()
    modelo = sm.OLS(datos.iloc[:, 0], datos.iloc[:, 1:]).fit()
    residuos_capm.loc[datos.index, cartera] = modelo.resid

residuos_capm = residuos_capm.dropna()
T_c = residuos_capm.shape[0]
N_c = residuos_capm.shape[1]
alpha_c = capm["alpha"].values.reshape(-1, 1)
Sigma_c = residuos_capm.cov().values
mkt_c = mkt_rf_solo.loc[residuos_capm.index]
SR2_c = (mkt_c.mean() / mkt_c.std()) ** 2
quad_c = (alpha_c.T @ np.linalg.inv(Sigma_c) @ alpha_c).item()
grs_capm = ((T_c - N_c - 1) / N_c) * quad_c / (1 + SR2_c)
p_capm = 1 - stats.f.cdf(grs_capm, dfn=N_c, dfd=T_c - N_c - 1)

print(f"  CAPM (1 factor):  GRS = {grs_capm:.4f}, p-valor = {p_capm:.6f}")
print(f"  FF3 (3 factores): GRS = {grs_stat:.4f}, p-valor = {grs_pvalue:.6f}")
print(f"\n  Reducción en el estadístico GRS: {(1 - grs_stat/grs_capm)*100:.1f}%")

# ========================================================================
# GUARDADO
# ========================================================================
gammas.to_pickle(os.path.join(CARPETA_PROCESADOS, "gammas_ff3.pkl"))
resumen.to_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3.pkl"))
resumen.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa_ff3.csv"))
comparacion.to_csv(os.path.join(CARPETA_RESULTADOS, "comparacion_gammas_primas_ff3.csv"), index=False)

print(f"\n✓ Resultados guardados en '{CARPETA_PROCESADOS}/'")
print(f"✓ Tablas CSV en '{CARPETA_RESULTADOS}/'")
