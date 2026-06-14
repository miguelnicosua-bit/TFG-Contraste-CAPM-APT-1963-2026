"""
Figura 1 del TFG: Frontera de mínima varianza, cartera tangente T y LMC.

Plano media-desviación típica (sigma, mu) con:
  - Hipérbola de la frontera de mínima varianza (rama eficiente e ineficiente).
  - Recta tangente a la hipérbola desde (0, rf): Línea del Mercado de Capitales.
  - Punto de tangencia T claramente destacado y etiquetado.
  - Ejes con etiquetas griegas μ y σ correctamente posicionadas.

Autor: Miguel Suárez Crespo
TFG - Valoración de Activos
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Carpeta donde se encuentra este script (las imágenes se guardarán aquí)
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Configuración para que el SVG mantenga el texto seleccionable
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# ============================================================
# PARÁMETROS DE LA HIPÉRBOLA (frontera de mínima varianza)
# ============================================================
# Hipérbola de la forma sigma^2 = a + b*(mu - mu0)^2, con vértice en
# el punto de mínima varianza global (sigma_min, mu_min).
mu_min = 0.08      # Rendimiento esperado del vértice (mínima varianza global)
sigma_min = 0.10   # Desviación típica mínima (vértice)
b = 3.0            # Apertura de la hipérbola
rf = 0.03          # Tipo libre de riesgo

# Eje de medias (rendimientos esperados)
mu_eje = np.linspace(-0.02, 0.22, 400)

# Hipérbola (rama eficiente y rama ineficiente)
sigma_hiperbola = np.sqrt(sigma_min**2 + b * (mu_eje - mu_min)**2)

# Separar parte eficiente (mu >= mu_min) y parte ineficiente (mu < mu_min)
mask_ef = mu_eje >= mu_min
mask_inef = mu_eje <= mu_min

# ============================================================
# CARTERA TANGENTE T: punto donde la recta desde (0, rf) toca la hipérbola
# ============================================================
# Para la hipérbola sigma^2 = sigma_min^2 + b*(mu - mu_min)^2, la condición de
# tangencia desde (0, rf) se obtiene maximizando la pendiente (mu - rf)/sigma.
# Resolviendo derivada igual a cero:
#   sigma_T^2 = sigma_min^2 + b*(mu_T - mu_min)^2
#   pendiente máxima -> mu_T = mu_min + sigma_min^2 / (b * (mu_min - rf))
mu_T = mu_min + sigma_min**2 / (b * (mu_min - rf))
sigma_T = np.sqrt(sigma_min**2 + b * (mu_T - mu_min)**2)

# Línea del Mercado de Capitales (LMC): recta desde (0, rf) que pasa por T
pendiente_LMC = (mu_T - rf) / sigma_T
sigma_LMC = np.linspace(0, 0.32, 100)
mu_LMC = rf + pendiente_LMC * sigma_LMC

# ============================================================
# REPRESENTACIÓN
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6.5))

# Hipérbola eficiente (parte superior, línea continua)
ax.plot(sigma_hiperbola[mask_ef], mu_eje[mask_ef],
        color='black', linewidth=2,
        label='Frontera eficiente')

# Hipérbola ineficiente (parte inferior, línea discontinua)
ax.plot(sigma_hiperbola[mask_inef], mu_eje[mask_inef],
        color='gray', linewidth=1.5, linestyle='--',
        label='Frontera ineficiente')

# LMC: tramo continuo (de rf a T, posiciones combinando activo libre + cartera T)
sigma_LMC_pre = np.linspace(0, sigma_T, 100)
mu_LMC_pre = rf + pendiente_LMC * sigma_LMC_pre
ax.plot(sigma_LMC_pre, mu_LMC_pre, color='C0', linewidth=2,
        label='Línea del Mercado de Capitales (LMC)')

# LMC: tramo discontinuo (más allá de T, posiciones apalancadas)
sigma_LMC_post = np.linspace(sigma_T, 0.32, 100)
mu_LMC_post = rf + pendiente_LMC * sigma_LMC_post
ax.plot(sigma_LMC_post, mu_LMC_post, color='C0', linewidth=2,
        linestyle='--', label='LMC apalancada')

# Cartera tangente T: punto destacado
ax.scatter([sigma_T], [mu_T], s=120, color='C3', zorder=5,
           edgecolors='black', linewidths=1.2)

# Etiqueta T con flecha (offset claro para que se vea, NO encima de la recta)
ax.annotate('T',
            xy=(sigma_T, mu_T),
            xytext=(sigma_T + 0.025, mu_T - 0.015),
            fontsize=18, fontweight='bold', color='C3',
            arrowprops=dict(arrowstyle='-', color='C3', lw=1))

# Punto del activo libre de riesgo
ax.scatter([0], [rf], s=80, color='C2', zorder=5,
           edgecolors='black', linewidths=1.0)
ax.annotate(r'$r_f$',
            xy=(0, rf),
            xytext=(-0.005, rf), 
            fontsize=14, color='C2', va='center', ha='right')

# ============================================================
# EJES Y ETIQUETAS
# ============================================================
ax.set_xlim(-0.005, 0.32)
ax.set_ylim(-0.01, 0.22)

# Cuadrícula sutil
ax.grid(True, alpha=0.25, linestyle=':')

# Ejes de coordenadas internos en X=0 e Y=0 (los únicos que se muestran)
ax.axhline(0, color='black', linewidth=1.0)
ax.axvline(0, color='black', linewidth=1.0)

# Ocultar todos los spines exteriores: solo quedan los ejes internos X=0 e Y=0
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Etiquetas de los ejes (griegas grandes y bien colocadas, NO en medio de las flechas)
ax.set_xlabel(r'Desviación típica  $\sigma$', fontsize=14, labelpad=-5)
ax.set_ylabel(r'Rendimiento esperado  $\mu$', fontsize=14, labelpad=-5)

# Quitar etiquetas numéricas si quieres versión "estilizada" para el TFG
# (déjalas si prefieres que se vean los valores; comentar las dos líneas siguientes
# si quieres mantener los valores numéricos)
ax.set_xticks([])
ax.set_yticks([])

# Leyenda en posición clara, abajo a la derecha
ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=11)

# Ajustes finales
plt.tight_layout()

# Guardar en SVG (vectorial, texto seleccionable) y en PNG (para previsualizar)
ruta_svg = os.path.join(CARPETA_SCRIPT, 'figura1_frontera_LMC.svg')
ruta_png = os.path.join(CARPETA_SCRIPT, 'figura1_frontera_LMC.png')
plt.savefig(ruta_svg, format='svg', bbox_inches='tight')
plt.savefig(ruta_png, format='png', dpi=200, bbox_inches='tight')

plt.show()

print(f"Figura generada en:\n  {ruta_svg}\n  {ruta_png}")
