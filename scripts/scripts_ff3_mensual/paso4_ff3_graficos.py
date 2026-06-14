"""
PASO 4 FF3 - Gráficos del contraste empírico del modelo de tres factores
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Genera seis figuras en formato vectorial SVG con texto seleccionable,
listas para insertar en el TFG:
  1. LMA empírica vs teórica (rendimiento medio vs beta_MKT FF3)
  2. Mapa de calor 10x10 de las betas de mercado (beta_MKT)
  3. Mapa de calor 10x10 de las betas SMB
  4. Mapa de calor 10x10 de las betas HML
  5. Mapa de calor 10x10 de los alfas FF3 (anualizados)
  6. Distribución de los t-estadísticos de los alfas FF3

Coherencia metodológica:
  - Los heatmaps y descriptivos (Gráficos 2-6) se construyen con la primera
    etapa ESTÁTICA (estimación con toda la muestra). Estos valores son
    descripciones transversales que NO dependen de la metodología rolling.
  - La LMA empírica (Gráfico 1) se construye con el coeficiente promedio
    gamma_MKT de la segunda etapa Fama-MacBeth rolling 60m.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Configuración para gráficos vectoriales con texto seleccionable
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from matplotlib.colors import TwoSlopeNorm

# ========================================================================
# CARGA DE DATOS
# ========================================================================
CARPETA_PROCESADOS = "datos_procesados"
CARPETA_GRAFICOS = "resultados/graficos/ff3_mensual"
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

primera      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa_ff3.pkl"))
factores_ff3 = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "factores_ff3.pkl"))
segunda      = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa_ff3.pkl"))

# Coeficiente promedio del mercado en Fama-MacBeth FF3 (anualizado, %)
g0_anual_ff3   = segunda.loc["gamma_0",   "media"] * 12 * 100
gMKT_anual_ff3 = segunda.loc["gamma_MKT", "media"] * 12 * 100

# Estilo general
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 100,
})

# ========================================================================
# FUNCIÓN AUXILIAR: PARSEAR NOMBRES DE CARTERAS
# ========================================================================
def parsear_nombre(nombre):
    n = nombre.strip()
    if n == "SMALL LoBM": return 1, 1
    if n == "SMALL HiBM": return 1, 10
    if n == "BIG LoBM":   return 10, 1
    if n == "BIG HiBM":   return 10, 10
    m = re.match(r"ME(\d+)\s+BM(\d+)", n)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Nombre no reconocido: {nombre}")

parsed = pd.DataFrame(
    [parsear_nombre(c) for c in primera.index],
    columns=["size", "bm"],
    index=primera.index
)
datos = pd.concat([primera, parsed], axis=1)

# ========================================================================
# GRÁFICO 1: LMA EMPÍRICA (FAMA-MACBETH ROLLING FF3) VS TEÓRICA
# ========================================================================
fig, ax = plt.subplots(figsize=(9, 6))

betas_MKT  = datos["beta_MKT"]
rend_anual = datos["rend_medio"] * 12 * 100
prima_mkt  = factores_ff3["Mkt-RF"].mean() * 12 * 100

sc = ax.scatter(betas_MKT, rend_anual, c=datos["bm"], cmap="RdYlGn",
                s=70, edgecolors="black", linewidths=0.4, alpha=0.85, zorder=3)

x_teo = np.linspace(0.5, 1.7, 100)

# LMA teórica del FF3 (idealmente la beta de mercado debería compensarse con la
# prima de mercado realizada)
y_teo = x_teo * prima_mkt
ax.plot(x_teo, y_teo, "--", color="black", linewidth=2,
        label=f"LMA teórica del CAPM/FF3 (pendiente = {prima_mkt:.2f}%)", zorder=2)

# LMA empírica: pendiente de gamma_MKT en Fama-MacBeth FF3
y_emp = g0_anual_ff3 + gMKT_anual_ff3 * x_teo
ax.plot(x_teo, y_emp, "-", color="red", linewidth=2,
        label=f"LMA empírica FF3 (pendiente = {gMKT_anual_ff3:.2f}%)", zorder=2)

# Cartera de mercado: estrella dorada en (beta=1, rendimiento=prima_mkt)
ax.scatter([1.0], [prima_mkt], marker="*", s=300, color="gold",
           edgecolors="black", linewidths=1.0, zorder=4, label="Cartera de mercado")

ax.set_xlabel("Beta de mercado estimada ($\\hat{\\beta}_{i,M}$, FF3)")
ax.set_ylabel("Exceso de rendimiento medio anualizado (%)")
ax.set_title("")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(sc, ax=ax, ticks=[1, 5, 10])
cbar.set_label("Decil de Book-to-Market\n(1 = growth, 10 = value)",
               rotation=270, labelpad=35)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "01_LMA_teorica_vs_empirica_FF3_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 1 guardado: 01_LMA_teorica_vs_empirica_FF3_mensual.svg")

# ========================================================================
# FUNCIÓN AUXILIAR PARA HEATMAPS 10x10
# ========================================================================
def heatmap_factor(columna_datos, titulo_archivo, etiqueta_cbar, cmap,
                   centrar_en=None, formato=":.2f"):
    """Heatmap 10x10 de una columna del DataFrame `datos`."""
    matriz = np.full((10, 10), np.nan)
    for _, fila in datos.iterrows():
        i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
        matriz[i, j] = fila[columna_datos]

    fig, ax = plt.subplots(figsize=(9, 7))

    if centrar_en is not None:
        vmax_d = max(abs(np.nanmax(matriz) - centrar_en), abs(np.nanmin(matriz) - centrar_en))
        norm = TwoSlopeNorm(vmin=centrar_en - vmax_d, vcenter=centrar_en, vmax=centrar_en + vmax_d)
        im = ax.imshow(matriz, cmap=cmap, norm=norm, aspect="equal")
        umbral_blanco = vmax_d * 0.55
    else:
        im = ax.imshow(matriz, cmap=cmap, aspect="equal")
        umbral_blanco = (np.nanmax(matriz) - np.nanmin(matriz)) * 0.6

    for i in range(10):
        for j in range(10):
            valor = matriz[i, j]
            if centrar_en is not None:
                dist = abs(valor - centrar_en)
                color_texto = "white" if dist > umbral_blanco else "black"
            else:
                color_texto = "white" if valor > np.nanmax(matriz) * 0.65 else "black"
            ax.text(j, i, f"{valor:{formato.lstrip(':')}}",
                    ha="center", va="center", fontsize=8, color=color_texto)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
    ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
    ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
    ax.set_ylabel("Decil Tamaño (1 = pequeña   ←→   10 = grande)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(etiqueta_cbar, rotation=270, labelpad=20)
    if centrar_en is not None:
        cbar.ax.axhline(centrar_en, color="black", linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, titulo_archivo),
                format="svg", bbox_inches="tight")
    plt.close()
    return matriz

# ========================================================================
# GRÁFICO 2: HEATMAP 10x10 DE LAS BETAS DE MERCADO (beta_MKT)
# ========================================================================
heatmap_factor("beta_MKT", "02_heatmap_beta_MKT_FF3_mensual.svg",
               "Beta de mercado estimada ($\\hat{\\beta}_{i,M}$)", "RdBu_r",
               centrar_en=1.0, formato=":.2f")
print("✓ Gráfico 2 guardado: 02_heatmap_beta_MKT_FF3_mensual.svg")

# ========================================================================
# GRÁFICO 3: HEATMAP 10x10 DE LAS BETAS SMB
# ========================================================================
heatmap_factor("beta_SMB", "03_heatmap_beta_SMB_FF3_mensual.svg",
               "Sensibilidad estimada al factor SMB ($\\hat{\\beta}_{i,SMB}$)",
               "RdBu_r", centrar_en=0.0, formato=":.2f")
print("✓ Gráfico 3 guardado: 03_heatmap_beta_SMB_FF3_mensual.svg")

# ========================================================================
# GRÁFICO 4: HEATMAP 10x10 DE LAS BETAS HML
# ========================================================================
heatmap_factor("beta_HML", "04_heatmap_beta_HML_FF3_mensual.svg",
               "Sensibilidad estimada al factor HML ($\\hat{\\beta}_{i,HML}$)",
               "RdBu_r", centrar_en=0.0, formato=":.2f")
print("✓ Gráfico 4 guardado: 04_heatmap_beta_HML_FF3_mensual.svg")

# ========================================================================
# GRÁFICO 5: HEATMAP 10x10 DE LOS ALFAS FF3 (anualizados, en %)
# ========================================================================
# Pasamos alpha mensual a alpha anualizado en %
datos["alpha_anual"] = datos["alpha"] * 12 * 100
heatmap_factor("alpha_anual", "05_heatmap_alfas_FF3_mensual.svg",
               "Alfa FF3 anualizado (%)", "RdYlGn",
               centrar_en=0.0, formato=":.1f")
print("✓ Gráfico 5 guardado: 05_heatmap_alfas_FF3_mensual.svg")

# ========================================================================
# GRÁFICO 6: DISTRIBUCIÓN DE LOS T-ESTADÍSTICOS DE LOS ALFAS FF3
# ========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

t_stats = datos["alpha_t"]
ax.hist(t_stats, bins=20, color="seagreen", edgecolor="black", alpha=0.75)

ax.axvline(-1.96, color="red", linestyle="--", linewidth=1.5,
           label="Umbrales $\\pm 1.96$ (significación al 5%)")
ax.axvline(1.96, color="red", linestyle="--", linewidth=1.5)
ax.axvline(0, color="black", linewidth=1)

n_sig_5 = (t_stats.abs() > 1.96).sum()
n_sig_1 = (t_stats.abs() > 2.58).sum()
texto = (f"Alfas FF3 significativos al 5%: {n_sig_5} / 100\n"
         f"Alfas FF3 significativos al 1%: {n_sig_1} / 100\n"
         f"Esperados bajo $H_0$ al 5%: ~5")
ax.text(0.02, 0.97, texto, transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="gray", alpha=0.9))

ax.set_xlabel("t-estadístico de $\\hat{\\alpha}_i$ (modelo FF3)")
ax.set_ylabel("Número de carteras")
ax.set_title("")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "06_tstats_alfas_FF3_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 6 guardado: 06_tstats_alfas_FF3_mensual.svg")

print(f"\n✓ Los 6 gráficos están en '{CARPETA_GRAFICOS}/'")
print("✓ Formato SVG vectorial con texto seleccionable")
