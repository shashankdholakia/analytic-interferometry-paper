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

load = Loader(paths.data)


HOUR_ANGLES = 50
#using SPICA lowres
WAVS = 1

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

veritas_tels = np.array([[140,-10],[50,-50],[40,60],[-30,10]])
station_x = veritas_tels[:,0]
station_y = veritas_tels[:,1]

baseline_inds, baselines = makebaselines(np.vstack([station_x, station_y]).T)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Veritas telescope positions")
plt.grid(True, linestyle='--', linewidth=0.5, color='k', alpha=0.5)
plt.scatter(station_x, station_y)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(paths.figures / "veritas_tels.pdf", dpi=300)

# ---------------------
# Set up the plot layout
fig = plt.figure(figsize=(11, 14))
fig.subplots_adjust(hspace=0.2, wspace=0.0)

# Mollweide map: use full width (colspan=8)
ax = [
    plt.subplot2grid((45, 8), (0, 0), rowspan=11, colspan=4,
                     projection='mollweide'),  # add projection here
    plt.subplot2grid((45, 8), (0, 5), rowspan=11, colspan=3)  # add projection here
]

# One row of 8 small orthographic views
ax_ortho = [
    plt.subplot2grid((45, 8), (14, n), rowspan=4, colspan=1) for n in range(8)
]

# Bottom panel for data
ax_data = [plt.subplot2grid((45, 8), (18, 0), rowspan=12, colspan=8)]


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



print("Loading earth data...")
ts = load.timescale()
    
t =ts.utc(2023, 12, 23, np.linspace(4,12,HOUR_ANGLES))
planets = load('de421.bsp')
earth = planets['earth']

print("Loading Hipparcos data on Alioth...")
with load.open(hipparcos.URL) as f:
    df = hipparcos.load_dataframe(f)

latitude = 31.66
veritas = earth + wgs84.latlon(latitude * N, 110.95 * W, elevation_m=1270)
alioth = Star.from_dataframe(df.loc[62956])
position = veritas.at(t).observe(alioth).apparent()

ha, dec, distance = position.hadec()
#matrix to project a star's changing hour angle and declination onto the baselines
#to create uv tracks
proj_mat = []
for h, d in zip(ha.radians, dec.radians):
    proj_mat.append(np.array([[np.sin(h), np.cos(h), 0],
                  [-np.sin(d)*np.cos(h), np.sin(d)*np.sin(h), np.cos(d)],
                  [np.cos(d)*np.cos(h), -np.cos(d)*np.sin(h), np.sin(d)]]))
proj_mat = np.array(proj_mat)

#project the baselines onto the uv plane
enu = np.insert(baselines, 2, 0, axis=1)
# Latitude in radians
latitude = np.deg2rad(latitude)  # example: 34 degrees

# Define the transformation matrix
T = np.array([
    [0, -np.sin(latitude), np.cos(latitude)],
    [1, 0, 0],
    [0, np.cos(latitude), np.sin(latitude)]
])

# Transform to (x, y, z)
xyz = enu @ T.T
wav = np.array([0.416*1e-6])
uv = (proj_mat @ xyz.T[None, :, :])[:,0:2, :]

#really complicated logic to first
#create a new axis for each wavelength
#then repeat the uv tracks for each wavelength
#then divide each uv track by the wavelength
#then transopse to get an array of (n_wavelengths, n_hourangles, 2, n_baselines)
uv_by_wav = (uv[np.newaxis,:,:].repeat(len(wav),axis=0).T/wav).T

u = np.concatenate(uv_by_wav[:,:,0],axis=0)
v = np.concatenate(uv_by_wav[:,:,1],axis=0)

print("Plotting uv coverage...")

ax[1].set_aspect("equal", adjustable="datalim")

print("u shape: " + str(u.shape))
print("v shape: " + str(v.shape))
#wavs = wav.repeat(HOUR_ANGLES,axis=0).repeat(u.shape[1], axis=0)
colors = plt.cm.tab20(np.linspace(0, 1, u.shape[1]))
for i in range(u.shape[1]):
    ax[1].scatter(u[:,i],v[:,i],color=colors[i],s=2.)
    ax[1].scatter(-u[:,i],-v[:,i],color=colors[i],s=2.)
ax[1].set_xlabel("U (baseline/$\lambda$)")
ax[1].set_ylabel("V (baseline/$\lambda$)")

radius = 1.47/2.
star_interferometry = Harmonix(star, radius)
vis_data = visibilities(star_interferometry, jnp.array(u.T), jnp.array(v.T), times)

print("Visibility data shape: " + str(vis_data.shape))
for n in range(8):
    for i in range(u.shape[1]):
        ax_data[0].plot(jnp.sqrt(u[:,i]**2+v[:,i]**2), vis_data[n,i,:], alpha=1.0,color=colors[i], lw=0.5, rasterized=True)
    

ax_data[0].set_xlabel('Spatial Frequency (lambdas)')
ax_data[0].set_xlim(left=0)
ax_data[0].set_ylim(bottom=0, top=1.0)
ax_data[0].set_ylabel('Visibility Amplitude')

plt.savefig(paths.figures / 'spot_map_veritas.pdf', bbox_inches="tight", dpi=300)