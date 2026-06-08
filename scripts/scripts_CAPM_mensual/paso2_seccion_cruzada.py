"""
PASO 2 - Segunda etapa Fama-MacBeth: regresión de sección cruzada
TFG Valoración de Activos - Contraste empírico del CAPM
Autor: Miguel Suárez Crespo

Para cada mes t, regresamos los excesos de rendimiento de las 100 carteras
sobre las betas estimadas en el Paso 1:

    r_{i,t} - r_{f,t} = gamma_{0,t} + gamma_{1,t} * beta_i_hat + u_{i,t}

Luego promediamos los T coeficientes obtenidos y aplicamos el contraste de
Fama-MacBeth (t-estadístico basado en la serie temporal de coeficientes).

Predicciones del CAPM:
  - gamma_0 = 0          (con excesos de rendimiento)
  - gamma_1 > 0 y aprox = E[r_M - r_f]
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
mkt_rf = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))
primera = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))

betas = primera["beta"]  # Serie con las 100 betas estimadas

print(f"Datos cargados:")
print(f"  - Excesos de rendimiento: {excesos.shape}")
print(f"  - Betas estimadas: {len(betas)} carteras")

# ========================================================================
# REGRESIONES DE SECCIÓN CRUZADA MES A MES
# ========================================================================
# Para cada mes t:
#   y_t = vector de excesos de las 100 carteras en el mes t
#   X = constante + betas (las mismas para todos los meses)
# Estimamos gamma_0,t y gamma_1,t por MCO.

gammas_0 = []  # Lista para almacenar gamma_0 de cada mes
gammas_1 = []  # Lista para almacenar gamma_1 de cada mes
fechas_t = []

X = sm.add_constant(betas)  # Diseño fijo: [1, beta_i]

for fecha in excesos.index:
    y = excesos.loc[fecha]  # Vector de 100 rendimientos en este mes

    # Algunas carteras pueden tener NaN en algún mes concreto.
    # Eliminamos esas observaciones puntualmente.
    datos = pd.concat([y, X], axis=1).dropna()
    if len(datos) < 10:
        continue  # Saltamos meses con muy pocos datos válidos
    y_clean = datos.iloc[:, 0]
    X_clean = datos.iloc[:, 1:]

    modelo = sm.OLS(y_clean, X_clean).fit()
    gammas_0.append(modelo.params["const"])
    gammas_1.append(modelo.params["beta"])
    fechas_t.append(fecha)

gammas = pd.DataFrame({
    "gamma_0": gammas_0,
    "gamma_1": gammas_1
}, index=fechas_t)

print(f"\nNúmero de regresiones de sección cruzada ejecutadas: {len(gammas)}")

# ========================================================================
# AGREGACIÓN: MEDIA TEMPORAL Y T-ESTADÍSTICOS (Fama-MacBeth)
# ========================================================================
# La media temporal de los gammas estima los parámetros poblacionales.
# La desviación típica de la serie temporal de gammas, dividida por sqrt(T),
# es el error estándar correcto (esto es la innovación de Fama-MacBeth).

T = len(gammas)

resumen = pd.DataFrame(index=["gamma_0", "gamma_1"])
resumen["media"] = gammas.mean().values
resumen["desv_tipica"] = gammas.std().values
resumen["error_estandar"] = resumen["desv_tipica"] / np.sqrt(T)
resumen["t_stat"] = resumen["media"] / resumen["error_estandar"]
resumen["p_valor"] = 2 * (1 - stats.t.cdf(resumen["t_stat"].abs(), df=T-1))

# Versión anualizada para interpretar económicamente
resumen["media_anual_%"] = resumen["media"] * 12 * 100

print("\n" + "=" * 70)
print("RESULTADOS DE LA SEGUNDA ETAPA (Fama-MacBeth)")
print("=" * 70)
print(resumen.round(5))

# ========================================================================
# CONTRASTE CONTRA LAS PREDICCIONES DEL CAPM
# ========================================================================
prima_mercado_realizada = mkt_rf.mean()  # E[r_M - r_f] observado

print("\n" + "=" * 70)
print("CONTRASTE FRENTE A LAS PREDICCIONES DEL CAPM")
print("=" * 70)

# Contraste 1: gamma_0 = 0
g0_media = resumen.loc["gamma_0", "media"]
g0_t = resumen.loc["gamma_0", "t_stat"]
g0_p = resumen.loc["gamma_0", "p_valor"]
print(f"\n1) Contraste H0: gamma_0 = 0")
print(f"   gamma_0 estimado: {g0_media*12*100:.3f}% anual")
print(f"   t-estadístico:    {g0_t:.3f}")
print(f"   p-valor:          {g0_p:.4f}")
if g0_p < 0.05:
    print(f"   → Se RECHAZA H0 al 5%. gamma_0 es significativamente distinto de 0.")
    print(f"     Esto contradice la predicción del CAPM.")
else:
    print(f"   → No se rechaza H0 al 5%. gamma_0 es compatible con 0.")

# Contraste 2: gamma_1 = 0 (¿la beta tiene precio?)
g1_media = resumen.loc["gamma_1", "media"]
g1_t = resumen.loc["gamma_1", "t_stat"]
g1_p = resumen.loc["gamma_1", "p_valor"]
print(f"\n2) Contraste H0: gamma_1 = 0  (¿la beta es precio del riesgo?)")
print(f"   gamma_1 estimado: {g1_media*12*100:.3f}% anual")
print(f"   t-estadístico:    {g1_t:.3f}")
print(f"   p-valor:          {g1_p:.4f}")
if g1_p < 0.05 and g1_media > 0:
    print(f"   → gamma_1 es significativamente POSITIVO. La beta sí tiene precio.")
elif g1_p < 0.05 and g1_media < 0:
    print(f"   → gamma_1 es significativamente NEGATIVO. Contradice el CAPM.")
else:
    print(f"   → gamma_1 NO es significativamente distinto de 0.")
    print(f"     La beta NO parece tener un precio de mercado positivo.")
    print(f"     Esto contradice la predicción central del CAPM.")

# Contraste 3: gamma_1 vs prima de mercado realizada
print(f"\n3) Comparación de gamma_1 con la prima de mercado observada:")
print(f"   gamma_1 estimado:               {g1_media*12*100:.3f}% anual")
print(f"   Prima de mercado E[r_M - r_f]:  {prima_mercado_realizada*12*100:.3f}% anual")
print(f"   Ratio gamma_1 / prima:          {g1_media/prima_mercado_realizada:.3f}")
print(f"   ")
print(f"   Bajo el CAPM ambas magnitudes deberían coincidir (ratio = 1).")

# ========================================================================
# GUARDADO
# ========================================================================
gammas.to_pickle(os.path.join(CARPETA_PROCESADOS, "gammas_mensuales.pkl"))
resumen.to_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa.pkl"))
resumen.to_csv(os.path.join(CARPETA_RESULTADOS, "segunda_etapa.csv"))

print(f"\n✓ Resultados guardados en '{CARPETA_PROCESADOS}/'")
print(f"✓ Tabla CSV exportada a '{CARPETA_RESULTADOS}/segunda_etapa.csv'")
print("✓ Listos para ejecutar paso3_contrastes.py")
