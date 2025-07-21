import jax
from jax import config
config.update("jax_enable_x64", True)  
import jax.numpy as jnp
from jax import jit, vmap, grad
import zodiax as zdx

import numpy as np
from harmonix.harmonix import Harmonix, visibilities, closure_phases
from harmonix.utils import maketriples_all, makebaselines
from jaxoplanet.starry import Surface, Ylm, show_surface
from jaxoplanet.starry.light_curves import surface_light_curve


import matplotlib.pyplot as plt
import paths
from functools import partial

from skyfield.api import load
from skyfield.api import Star
from skyfield.api import Loader

from skyfield.data import hipparcos
from skyfield.api import N,S,E,W, wgs84
plt.rcParams.update({'font.size': 12})

# Assume you have a Surface object with an intensity method
# For example:
y_star = np.load(paths.data / "SPOT_map_highres.npy")
y = Ylm.from_dense(y_star)
star = Surface(y=y, inc=jnp.radians(60.), obl=0, period=1.0)

# 1. Create longitude and latitude meshgrid
n_lon = 360*2
n_lat = 180*2

lon = jnp.linspace(-jnp.pi, jnp.pi, n_lon)
lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_lat)
lon_grid, lat_grid = jnp.meshgrid(lon, lat)

# 2. Compute intensity at each (lat, lon)
intensity = star.intensity(lat_grid, lon_grid)[::-1, ::-1]


# ---------------------
# Set up the plot layout
fig = plt.figure(figsize=(11, 9))
fig.subplots_adjust(hspace=0.2, wspace=0.0)

# Mollweide map: use full width (colspan=8)
ax = [
    plt.subplot2grid((33, 8), (0, 0), rowspan=11, colspan=5,
                     projection='mollweide'),  # add projection here
    plt.subplot2grid((33, 8), (0, 5), rowspan=11, colspan=3)  # add projection here
]
# One row of 8 small orthographic views
ax_ortho = [
    plt.subplot2grid((33, 8), (14, n), rowspan=4, colspan=1) for n in range(8)
]

# Bottom panel for data
ax_data = plt.subplot2grid((33, 8), (18, 0), rowspan=14, colspan=8)

# ---------------------
# Plot the Mollweide map
pcm = ax[0].pcolormesh(np.asarray(lon_grid), np.asarray(lat_grid), np.asarray(intensity),
                       shading='auto', cmap='plasma', rasterized=True)
#ax[0].set_title("Surface Intensity Map (Mollweide Projection)")
ax[0].set_longitude_grid_ends(90)
ax[0].set_longitude_grid(60)
ax[0].set_latitude_grid(30)
ax[0].grid(True, linestyle='-', linewidth=0.5, color='k', alpha=0.3)
ax[0].tick_params(axis='x', labelbottom=False) # Hide x-axis tick labels
ax[0].tick_params(axis='y', labelleft=False)   # Hide y-axis tick labels
#fig.colorbar(pcm, ax=ax[0], orientation='vertical')

# ---------------------
# (Optional) Fill in ax_ortho and ax_data with your content later
times = jnp.linspace(0, 1, 8)  # Example times for the orthographic views
for n in range(8):
    show_surface(star, ax=ax_ortho[n], cmap='plasma', theta=star.rotational_phase(times[n]))

u, v = np.zeros((250,1)), jnp.array([np.linspace(-330,330,250)]).T
u1, v1 = jnp.array([np.linspace(-330,330,250)]).T, np.zeros((250,1))
ax[1].scatter(u, v, s=0.5, color='k', rasterized=True)
ax[1].scatter(u1, v1, s=0.5, color='b', rasterized=True)
ax[1].set_xlabel('u (m)')
ax[1].set_ylabel('v (m)')
ax[1].set_aspect('equal', adjustable='box')
wavel = 7e-7 # m
radius = 1.47/2.
star_interferometry = Harmonix(star, radius)
vis_data = visibilities(star_interferometry, u/wavel, v/wavel, times)
vis_data_1 = visibilities(star_interferometry, u1/wavel, v1/wavel, times)
for n in range(8):
    ax_data.plot(jnp.sqrt(u**2+v**2), vis_data[n], alpha=0.3,color='k', lw=0.5, rasterized=True)
    ax_data.plot(jnp.sqrt(u1**2+v1**2), vis_data_1[n], alpha=0.3,color='b', lw=0.5, rasterized=True)
ax_data.set_xlabel('Baseline (m)')
ax_data.set_ylabel('Visibility Amplitude')
plt.savefig(paths.figures / 'spot_map.pdf', bbox_inches="tight", dpi=300)