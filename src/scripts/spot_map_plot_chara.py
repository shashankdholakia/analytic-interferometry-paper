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


HOUR_ANGLES = 1
#using SPICA lowres
WAVS = 50

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

ax_data.append(plt.subplot2grid((45, 8), (32, 0), rowspan=12, colspan=8))

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

chara_tels = np.array(
    [[0, 0],
    [330.66,22.28],
    [-313.53,253.39],
    [302.33,25.7],
    [-221.82,241.27],
    [-65.88,236.6]])
#for some reason the file is rotated by 90 degrees
theta = -90
chara_tels[:,1] = chara_tels[:,1] + theta
station_x = chara_tels[:,0]*np.cos(np.radians(chara_tels[:,1]))
station_y = chara_tels[:,0]*np.sin(np.radians(chara_tels[:,1]))
station_x-=np.abs(station_x.min())
station_y+=np.abs(station_y.min())

cp_inds, cp_uvs = maketriples_all(np.vstack([station_x, station_y]).T)[0:10]
print("cp_inds shape: " + str(cp_inds.shape))
baseline_inds, baselines = makebaselines(np.vstack([station_x, station_y]).T)

print("cp_uvs shape: " + str(cp_uvs.shape))

print("Loading earth data...")
ts = load.timescale()
    
t =ts.utc(2023, 3, 23, np.linspace(8,12,HOUR_ANGLES))
planets = load('de421.bsp')
earth = planets['earth']

print("Loading Hipparcos data on Alioth...")
with load.open(hipparcos.URL) as f:
    df = hipparcos.load_dataframe(f)

chara = earth + wgs84.latlon(34.2249 * N, 118.0564 * W, elevation_m=1740)
alioth = Star.from_dataframe(df.loc[62956])
position = chara.at(t).observe(alioth).apparent()

ha, dec, distance = position.hadec()

#matrix to project a star's changing hour angle and declination onto the baselines
#to create uv tracks
proj_mat = []
for h, d in zip(ha.radians, dec.radians):
    proj_mat.append(np.array([[np.sin(h), np.cos(h), 0],
                  [-np.sin(d)*np.cos(h), np.sin(d)*np.sin(h), np.cos(d)],
                  [np.cos(d)*np.cos(h), -np.cos(d)*np.sin(h), np.sin(d)]]))
proj_mat = np.array(proj_mat)
proj_mat.shape

#project the baselines onto the uv plane
xyz = np.insert(baselines, 2, 0, axis=1)
wav = jnp.linspace(0.65*1e-6, 0.95*1e-6,WAVS)
uv = (proj_mat@xyz.T)[:,0:2]
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
    ax[1].scatter(u[:,i],v[:,i],color=colors[i],s=1.)
ax[1].set_xlabel("U (baseline/$\lambda$)")
ax[1].set_ylabel("V (baseline/$\lambda$)")

radius = 1.47/2.
star_interferometry = Harmonix(star, radius)
vis_data = visibilities(star_interferometry, jnp.array(u.T), jnp.array(v.T), times)
cp_data = closure_phases(star_interferometry, jnp.array(u.T), jnp.array(v.T),times, cp_inds[0:10,0], cp_inds[0:10,1], cp_inds[0:10,2])
print("Visibility data shape: " + str(vis_data.shape))
for n in range(8):
    for i in range(u.shape[1]):
        ax_data[0].plot(jnp.sqrt(u[:,i]**2+v[:,i]**2), vis_data[n,i,:], alpha=1.0,color=colors[i], lw=0.5, rasterized=True)
    
    #get the maximum baseline for each closure phase
    cp_x_axis = []
    cp_x_color = []
    for i in range(10):
        #each station in the closure phase triple
        b1 = cp_inds[i,0]
        b2 = cp_inds[i,1]
        b3 = cp_inds[i,2]
        #each baseline in the closure phase triple, and its respectice index in u, v
        matches_1 = np.all(baseline_inds == [b1,b2], axis=1)
        ind1 = np.where(matches_1)[0]
        matches_2 = np.all(baseline_inds == [b2,b3], axis=1)
        ind2 = np.where(matches_2)[0]
        matches_3 = np.all(baseline_inds == [b1,b3], axis=1)
        ind3 = np.where(matches_3)[0]
        #and the index of the closure phase triple with the maximum baseline
        #the same index is repeated n_wavs times, so just take the first one
        max_baseline_ind = jnp.argmax(
            jnp.array([jnp.sqrt((jnp.array(u.T))**2+(jnp.array(v.T))**2)[ind1],
                       jnp.sqrt((jnp.array(u.T))**2+(jnp.array(v.T))**2)[ind2],
                       jnp.sqrt((jnp.array(u.T))**2+(jnp.array(v.T))**2)[ind3]]), axis=0)[0,0]
        if max_baseline_ind == 0:
            cp_x_axis.append(jnp.sqrt((jnp.array(u.T))**2+(jnp.array(v.T))**2)[ind1])
            cp_x_color.append(ind1)
        elif max_baseline_ind == 1:
            cp_x_axis.append(jnp.sqrt((jnp.array(u.T))**2+(jnp.array(v.T))**2)[ind2])
            cp_x_color.append(ind2)
        elif max_baseline_ind == 2:
            cp_x_axis.append(jnp.sqrt((jnp.array(u.T))**2+(jnp.array(v.T))**2)[ind3])
            cp_x_color.append(ind3)
        
    cp_x_color = np.array(cp_x_color)
    for i in range(len(cp_x_color)):
        ax_data[1].scatter(cp_x_axis[i], cp_data[n,i,:], color=colors[cp_x_color[i]], rasterized=True, s=1)

ax_data[0].set_xlabel('Spatial Frequency (lambdas)')
ax_data[0].set_ylabel('Visibility Amplitude')
ax_data[1].set_xlabel("spatial frequency at max baseline (lambdas)")
ax_data[1].set_ylabel("closure phase")

plt.savefig(paths.figures / 'spot_map_chara.pdf', bbox_inches="tight", dpi=300)