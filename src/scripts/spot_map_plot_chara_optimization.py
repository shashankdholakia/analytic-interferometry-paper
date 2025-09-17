import jax
from jax import config
config.update("jax_enable_x64", True)  
import jax.numpy as jnp
from jax import jit, vmap, grad
import jax.random as jr

import zodiax as zdx

import numpy as np
from harmonix.harmonix import Harmonix, visibilities, closure_phases
from harmonix.utils import maketriples_all, makebaselines
from jaxoplanet.starry import Surface, Ylm, show_surface
from jaxoplanet.starry.light_curves import surface_light_curve

import optax


import matplotlib.pyplot as plt
import paths
from functools import partial
from tqdm import tqdm

from skyfield.api import load
from skyfield.api import Star
from skyfield.api import Loader

from skyfield.data import hipparcos
from skyfield.api import N,S,E,W, wgs84
plt.rcParams.update({'font.size': 12})

load = Loader(paths.data)


HOUR_ANGLES = 5
#using SPICA lowres
WAVS = 50

PERIOD = 5.1

SNR = [20, 50, 100, 200, 500, 1000]

opt_params = ["data"]


def angular_mse(predicted, target):
    delta = (predicted - target + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return jnp.mean(delta**2)

@zdx.filter_jit
@zdx.filter_value_and_grad(opt_params)
def loss_fn(model, vis_data, cp_data, u, v, times, index_cps1, index_cps2, index_cps3):
    model_vis = visibilities(model, jnp.array(u.T), jnp.array(v.T), times)
    model_cp = closure_phases(model, jnp.array(u.T), jnp.array(v.T), times, index_cps1, index_cps2, index_cps3)
    #scale the noise in the closure phases by 360 degrees
    #gaussian prior penalty on the coefficients
    return (jnp.square((model_vis - vis_data)).mean() + angular_mse(jnp.radians(model_cp), jnp.radians(cp_data)))
    #return (jnp.square((model_cp - cp_data) / (360)).mean())


ncols = 3
nrows = (len(SNR) // ncols) + 1  # +1 for the true image on top

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, 
    subplot_kw={'projection': 'mollweide'}, 
    figsize=(5 * ncols, 3.5 * nrows)
)

axes = axes.flatten()

center_idx = ncols // 2

for i in range(ncols):
    if i != center_idx:
        axes[i].axis('off')
# Assume you have a Surface object with an intensity method
# For example:
y_star = np.load(paths.data / "SPOT_map_highres.npy")
y = Ylm.from_dense(y_star)
star = Surface(y=y, inc=jnp.radians(60.), obl=0, period=PERIOD)

# 1. Create longitude and latitude meshgrid
n_lon = 360*2
n_lat = 180*2

lon = jnp.linspace(-jnp.pi, jnp.pi, n_lon)
lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_lat)
lon_grid, lat_grid = jnp.meshgrid(lon, lat)

# 2. Compute intensity at each (lat, lon)
intensity = star.intensity(lat_grid, lon_grid)[::-1, ::-1]

if ncols % 2 == 1:
    ax = axes[center_idx]
    pcm = ax.pcolormesh(np.asarray(lon_grid), np.asarray(lat_grid), np.asarray(intensity),
                        shading='auto', cmap='plasma', rasterized=True)
    ax.set_longitude_grid_ends(90)
    ax.set_longitude_grid(60)
    ax.set_latitude_grid(30)
    ax.grid(True, linestyle='-', linewidth=0.5, color='k', alpha=0.3)
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)
    ax.set_title("True Image", fontsize=14)
    


chara_tels = np.array(
    [[0, 0],
    [330.66,22.28],
    [-313.53,253.39],
    [302.33,25.7],
    [-221.82,241.27],
    [-65.88,236.6]])
#for some reason the file is rotated by 90 degrees
theta = 0
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

latitude = 34.2249
chara = earth + wgs84.latlon(latitude * N, 118.0564 * W, elevation_m=1740)
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

times = jnp.linspace(0, PERIOD, 8, endpoint=False)  # Example times for the orthographic views

#subdivide the times further
window_size = 4/24 # how many hours per night
sub_offsets = np.linspace(-window_size/2, window_size/2, HOUR_ANGLES)
times = jnp.array([time + sub_offsets for time in times]).flatten()
radius = 1.47/2.
star_interferometry = Harmonix(star, radius)

vis_true = visibilities(star_interferometry, jnp.array(u.T), jnp.array(v.T), times)
cp_true = closure_phases(star_interferometry, jnp.array(u.T), jnp.array(v.T),times, cp_inds[0:10,0], cp_inds[0:10,1], cp_inds[0:10,2])


for i, snr in enumerate(SNR):
    ax = axes[ncols + i]
    key = jr.PRNGKey(0)
    vis_data = vis_true + jr.normal(key, vis_true.shape)/snr
    cp_data = cp_true + jr.normal(key, cp_true.shape)*360./snr

    # Now lets construct a loss function
    l_max = 15
    n_max = l_max**2 + 2 * l_max + 1
    y_star = np.zeros(n_max)
    y_star[0] = 1.0  # Set the zeroth coefficient to 1 for a constant term
    y_star = jnp.array(y_star, dtype=jnp.float64)  # Ensure y_star is a JAX array
    y = Ylm.from_dense(y_star)
    star = Surface(y=y, inc=jnp.radians(60.), obl=0, period=PERIOD)
    model = Harmonix(star, radius)
    model_init = model

    # Evaluate loss function once Compile to XLA
    print("Evaluating loss function once to compile for SNR =", snr)
    loss, grads = loss_fn(model, vis_data, cp_data, u, v, times, cp_inds[0:10,0], cp_inds[0:10,1], cp_inds[0:10,2])
    print("Loss function evaluated.")
    
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1e-4)
    )

    # Get optax objcets
    optimiser, state = zdx.get_optimiser(model, opt_params, opt)

    losses = []
    
    print("Starting optimization for SNR =", snr)
    for i in tqdm(range(6000)):
        loss, grads = loss_fn(model, vis_data, cp_data, u, v, times, cp_inds[0:10,0], cp_inds[0:10,1], cp_inds[0:10,2])
        step, state = optimiser.update(grads, state, params=model)
        model = zdx.apply_updates(model, step)
        losses.append(loss)
    print("Optimization completed for SNR =", snr)

    # 1. Create longitude and latitude meshgrid
    n_lon = 360*2
    n_lat = 180*2

    lon = jnp.linspace(-jnp.pi, jnp.pi, n_lon)
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_lat)
    lon_grid, lat_grid = jnp.meshgrid(lon, lat)

    y_star = jnp.concatenate([jnp.array([1.0]), model.data])  # Ensure y_star is a JAX array
    y = Ylm.from_dense(y_star)
    star = Surface(y=y, inc=jnp.radians(60.), obl=0, period=PERIOD, normalize=True)
    model_opt = Harmonix(star, radius)
    # 2. Compute intensity at each (lat, lon)
    intensity = model_opt.surface.intensity(lat_grid, lon_grid)[::-1, ::-1]
    # ---------------------
    # Plot the Mollweide map
    pcm = ax.pcolormesh(np.asarray(lon_grid), np.asarray(lat_grid), np.asarray(intensity),
                        shading='auto', cmap='plasma', rasterized=True)
    #ax[0].set_title("Surface Intensity Map (Mollweide Projection)")
    ax.set_longitude_grid_ends(90)
    ax.set_longitude_grid(60)
    ax.set_latitude_grid(30)
    ax.grid(True, linestyle='-', linewidth=0.5, color='k', alpha=0.3)
    ax.tick_params(axis='x', labelbottom=False) # Hide x-axis tick labels
    ax.tick_params(axis='y', labelleft=False)   # Hide y-axis tick labels
    ax.set_title(f"SNR={snr}", fontsize=14)
    

plt.savefig(paths.figures / 'chara_optimization.pdf', bbox_inches='tight', dpi=300)