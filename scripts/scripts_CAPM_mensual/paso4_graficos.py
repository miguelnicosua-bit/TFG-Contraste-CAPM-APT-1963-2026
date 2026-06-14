"""
PASO 4 - Gráficos del contraste empírico del CAPM
TFG Valoración de Activos
Autor: Miguel Suárez Crespo

Genera cinco figuras en formato vectorial SVG con texto seleccionable,
listas para insertar en el TFG:
  1. LMA empírica vs teórica
  2. Mapa de calor 10x10 de las betas estimadas
  3. Mapa de calor 10x10 de los alfas (anualizados)
  4. Mapa de calor 10x10 de los rendimientos medios (anualizados)
  5. Distribución de los t-estadísticos de los alfas

Coherencia metodológica:
  - Los heatmaps y descriptivos (Gráficos 2, 3, 4, 5) se construyen con
    la primera etapa ESTÁTICA (estimación con toda la muestra). Estos
    valores son descripciones transversales que NO dependen de la
    metodología rolling de la segunda etapa.
  - La LMA empírica (Gráfico 1) se construye con los coeficientes promedio
    gamma_0_bar y gamma_1_bar de la segunda etapa Fama-MacBeth con ventana
    móvil de 60 meses, que es la pendiente empírica resultante de la
    metodología contrastada.
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
CARPETA_GRAFICOS = "resultados/graficos/capm_mensual"
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

primera = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "primera_etapa.pkl"))
mkt_rf  = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "mkt_rf.pkl"))
segunda = pd.read_pickle(os.path.join(CARPETA_PROCESADOS, "segunda_etapa.pkl"))

# Coeficientes promedio de Fama-MacBeth rolling (anualizados, en %)
g0_anual = segunda.loc["gamma_0", "media"] * 12 * 100
g1_anual = segunda.loc["gamma_1", "media"] * 12 * 100

# Estilo general
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 100,
})

# ========================================================================
# FUNCIÓN AUXILIAR: PARSEAR NOMBRES DE CARTERAS A COORDENADAS (size, bm)
# ========================================================================
def parsear_nombre(nombre):
    """
    Convierte el nombre de una cartera a (decil_size, decil_bm), ambos 1-10.
    """
    n = nombre.strip()
    if n == "SMALL LoBM": return 1, 1
    if n == "SMALL HiBM": return 1, 10
    if n == "BIG LoBM":   return 10, 1
    if n == "BIG HiBM":   return 10, 10
    m = re.match(r"ME(\d+)\s+BM(\d+)", n)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Nombre no reconocido: {nombre}")

# DataFrame con coordenadas size, bm y demás datos
parsed = pd.DataFrame(
    [parsear_nombre(c) for c in primera.index],
    columns=["size", "bm"],
    index=primera.index
)
datos = pd.concat([primera, parsed], axis=1)

# ========================================================================
# GRÁFICO 1: LMA EMPÍRICA (FAMA-MACBETH ROLLING) VS TEÓRICA
# ========================================================================
fig, ax = plt.subplots(figsize=(9, 6))

betas = datos["beta"]
rend_anual = datos["rend_medio"] * 12 * 100
prima_mkt = mkt_rf.mean() * 12 * 100

sc = ax.scatter(betas, rend_anual, c=datos["bm"], cmap="RdYlGn",
                s=70, edgecolors="black", linewidths=0.4, alpha=0.85, zorder=3)

x_teo = np.linspace(0.5, 1.7, 100)

# LMA TEÓRICA del CAPM: y = prima_mkt * beta  (intercepto 0, pendiente = prima)
y_teo = x_teo * prima_mkt
ax.plot(x_teo, y_teo, "--", color="black", linewidth=2,
        label=f"LMA teórica del CAPM (pendiente = {prima_mkt:.2f}%)", zorder=2)

# LMA EMPÍRICA de Fama-MacBeth rolling 60m: y = g0_anual + g1_anual * beta
y_emp = g0_anual + g1_anual * x_teo
ax.plot(x_teo, y_emp, "-", color="red", linewidth=2,
        label=f"LMA empírica (pendiente = {g1_anual:.2f}%)", zorder=2)

# Cartera de mercado: estrella dorada en (beta=1, rendimiento=prima_mkt)
ax.scatter([1.0], [prima_mkt], marker="*", s=300, color="gold",
           edgecolors="black", linewidths=1.0, zorder=4, label="Cartera de mercado")

ax.set_xlabel("Beta estimada ($\\hat{\\beta}_i$)")
ax.set_ylabel("Exceso de rendimiento medio anualizado (%)")
ax.set_title("")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(sc, ax=ax, ticks=[1, 5, 10])
cbar.set_label("Decil de Book-to-Market\n(1 = growth, 10 = value)",
               rotation=270, labelpad=35)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "01_LMA_teorica_vs_empirica_CAPM_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 1 guardado: 01_LMA_teorica_vs_empirica_CAPM_mensual.svg")

# ========================================================================
# GRÁFICO 2: HEATMAP 10x10 DE LAS BETAS ESTIMADAS
# ========================================================================
matriz_beta = np.full((10, 10), np.nan)
for _, fila in datos.iterrows():
    i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
    matriz_beta[i, j] = fila["beta"]

fig, ax = plt.subplots(figsize=(9, 7))
# Centrado en 1 (beta del mercado) para que el color refleje agresivo/defensivo
vmax_b = max(abs(np.nanmax(matriz_beta) - 1), abs(np.nanmin(matriz_beta) - 1))
norm_b = TwoSlopeNorm(vmin=1 - vmax_b, vcenter=1, vmax=1 + vmax_b)
im = ax.imshow(matriz_beta, cmap="RdBu_r", norm=norm_b, aspect="equal")

# Anotamos cada celda con el valor numérico de la beta
for i in range(10):
    for j in range(10):
        valor = matriz_beta[i, j]
        dist = abs(valor - 1)
        color_texto = "white" if dist > vmax_b * 0.55 else "black"
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = pequeña   ←→   10 = grande)")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Beta estimada", rotation=270, labelpad=20)
cbar.ax.axhline(1, color="black", linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "02_heatmap_betas_CAPM_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 2 guardado: 02_heatmap_betas_CAPM_mensual.svg")

# ========================================================================
# GRÁFICO 3: HEATMAP 10x10 DE LOS ALFAS
# ========================================================================
matriz_alfa = np.full((10, 10), np.nan)
for _, fila in datos.iterrows():
    i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
    matriz_alfa[i, j] = fila["alpha"] * 12 * 100

fig, ax = plt.subplots(figsize=(9, 7))
vmax = np.nanmax(np.abs(matriz_alfa))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(matriz_alfa, cmap="RdYlGn", norm=norm, aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_alfa[i, j]
        color_texto = "white" if abs(valor) > vmax * 0.6 else "black"
        ax.text(j, i, f"{valor:.1f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = pequeña   ←→   10 = grande)")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Alfa anualizado (%)", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "03_heatmap_alfas_CAPM_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 3 guardado: 03_heatmap_alfas_CAPM_mensual.svg")

# ========================================================================
# GRÁFICO 4: HEATMAP 10x10 DE RENDIMIENTOS MEDIOS
# ========================================================================
matriz_rend = np.full((10, 10), np.nan)
for _, fila in datos.iterrows():
    i, j = int(fila["size"]) - 1, int(fila["bm"]) - 1
    matriz_rend[i, j] = fila["rend_medio"] * 12 * 100

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(matriz_rend, cmap="YlOrRd", aspect="equal")

for i in range(10):
    for j in range(10):
        valor = matriz_rend[i, j]
        max_val = np.nanmax(matriz_rend)
        color_texto = "white" if valor > max_val * 0.65 else "black"
        ax.text(j, i, f"{valor:.1f}", ha="center", va="center",
                fontsize=8, color=color_texto)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f"BM{j+1}" for j in range(10)])
ax.set_yticklabels([f"ME{i+1}" for i in range(10)])
ax.set_xlabel("Decil Book-to-Market (1 = growth   ←→   10 = value)")
ax.set_ylabel("Decil Tamaño (1 = pequeña   ←→   10 = grande)")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Rendimiento medio en exceso (%)", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "04_heatmap_rendimientos_CAPM_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 4 guardado: 04_heatmap_rendimientos_CAPM_mensual.svg")

# ========================================================================
# GRÁFICO 5: DISTRIBUCIÓN DE LOS T-ESTADÍSTICOS DE LOS ALFAS
# ========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

t_stats = datos["alpha_t"]
ax.hist(t_stats, bins=20, color="steelblue", edgecolor="black", alpha=0.75)

ax.axvline(-1.96, color="red", linestyle="--", linewidth=1.5,
           label="Umbrales $\\pm 1.96$ (significación al 5%)")
ax.axvline(1.96, color="red", linestyle="--", linewidth=1.5)
ax.axvline(0, color="black", linewidth=1)

n_sig_5 = (t_stats.abs() > 1.96).sum()
n_sig_1 = (t_stats.abs() > 2.58).sum()
texto = (f"Alfas significativos al 5%: {n_sig_5} / 100\n"
         f"Alfas significativos al 1%: {n_sig_1} / 100\n"
         f"Esperados bajo $H_0$ al 5%: ~5")
ax.text(0.02, 0.97, texto, transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="gray", alpha=0.9))

ax.set_xlabel("t-estadístico de $\\hat{\\alpha}_i$")
ax.set_ylabel("Número de carteras")
ax.set_title("")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, "05_tstats_alfas_CAPM_mensual.svg"),
            format="svg", bbox_inches="tight")
plt.close()
print("✓ Gráfico 5 guardado: 05_tstats_alfas_CAPM_mensual.svg")

print(f"\n✓ Los 5 gráficos están en '{CARPETA_GRAFICOS}/'")
print("✓ Formato SVG vectorial con texto seleccionable")
