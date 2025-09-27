import jax
from jax import config
config.update("jax_enable_x64", True)  
import jax.numpy as jnp
from jax import jit, vmap, grad
import jax.random as jr

import zodiax as zdx

import numpy as np
from harmonix.harmonix import Harmonix, visibilities
from harmonix.utils import maketriples_all, makebaselines
from jaxoplanet.starry import Surface, Ylm, show_surface
from jaxoplanet.starry.core.rotation import left_project
from jaxoplanet.starry.utils import y1d_to_2d, y2d_to_1d, C
from jaxoplanet.starry.light_curves import surface_light_curve

import s2fft
from s2fft.utils.quadrature_jax import quad_weights_mw_theta_only
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

PERIOD = 5.1  # Rotation period of the star in days
ROTATIONAL_PHASES = 8  # Number of rotational phases to consider

HOUR_ANGLES = 50
#using SPICA lowres
WAVS = 1


opt_params = ["data"]

### Spherical Harmonic Transform Utilities with S2FFT

def ylm_to_pixels(ylm_map, lmax):
    ylm_map_rot = left_project(lmax, 0.0, 0.0, 0.0, 0.0, ylm_map)
    ylm_2d = y1d_to_2d(lmax, ylm_map_rot)@C(lmax)
    pixels = s2fft.inverse_jax(ylm_2d, lmax+1, reality=True)
    return pixels

def pixels_to_ylm(pixel_map, lmax):
    ylm_2d = s2fft.forward_jax(pixel_map, lmax+1, reality=True)
    return ylm_2d@jnp.linalg.inv(C(lmax))

def smooth_abs(x, eps=1e-6):
    # differentiable |x|
    return jnp.sqrt(x*x + eps)

def tv_l1_penalty(pix, lam_tv=1e-7, lmax=15):
    w = quad_weights_mw_theta_only(lmax+1)
    pix = (pix.T/w).T
    # wrap in longitude; no wrap in latitude
    diff_lon = pix - jnp.roll(pix, shift=1, axis=1)
    diff_lat = pix[:-1, :] - pix[1:, :]
    tv = jnp.sum(smooth_abs(diff_lon)) + jnp.sum(smooth_abs(diff_lat))
    #tv = jnp.sum(smooth_abs(diff_lon)) + jnp.sum(smooth_abs(diff_lat))
    return lam_tv * tv

@zdx.filter_jit
def lc_func(model, t):
    theta = model.rotational_phase(t)
    y = Ylm.from_dense(jnp.concatenate([jnp.array([1.0]), model.data]))
    star = Surface(y=y, inc=model.surface.inc, obl=model.surface.obl, period=model.surface.period, u=model.surface.u)
    light_curve = vmap(partial(surface_light_curve, star, r=0., x=1., y=1., z=1.))(theta=theta)
    return light_curve

@zdx.filter_jit
@zdx.filter_value_and_grad(opt_params)
def loss_fn(model, vis_data, u, v, times, times_lc, lc_data, lmax=15):
    """
    Loss function for the visibility amplitude + light curve from 
    an intensity interferometer. 
    
    """
    model_vis = visibilities(model, jnp.array(u.T), jnp.array(v.T), times) 
    model_lc = lc_func(model, times_lc)
    
    pixels = ylm_to_pixels(jnp.concatenate([jnp.array([1.0]), model.data]), lmax)
    tv_penalty = tv_l1_penalty(pixels)
    return (jnp.square((model_vis - vis_data)).mean()) + jnp.mean((model_lc - lc_data) ** 2) + tv_penalty

ncols = 2
nrows = (2 // ncols) + 1  # +1 for the true image on top

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

    

#################################
# CTA North Simulation
#################################

print("Setting up CTA North Simulation")
cta_north_tels = np.loadtxt(paths.data / "CTA_North_positions_omega.csv", delimiter=",", skiprows=1)
station_x = cta_north_tels[:,0]
station_y = cta_north_tels[:,1]
baseline_inds, baselines = makebaselines(np.vstack([station_x, station_y]).T)

print("Loading earth data...")
ts = load.timescale()
    
t =ts.utc(2023, 3, 23, np.linspace(8,12,HOUR_ANGLES))
planets = load('de421.bsp')
earth = planets['earth']

print("Loading Hipparcos data on Alioth...")
with load.open(hipparcos.URL) as f:
    df = hipparcos.load_dataframe(f)

latitude = 28.7134
cta_north = earth + wgs84.latlon(latitude * N, 118.0564 * W, elevation_m=1740)
alioth = Star.from_dataframe(df.loc[62956])
position = cta_north.at(t).observe(alioth).apparent()

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

print(u.shape, v.shape)
fig1, ax = plt.subplots()
ax.set_aspect("equal", adjustable="datalim")
ax.scatter(u,v, c='k',s=1.);
ax.set_xlabel("U (lambdas)")
ax.set_ylabel("V (lambdas)")
plt.savefig(paths.figures / "uv_coverage_cta_north.png", dpi=300)

times = jnp.linspace(0,PERIOD,ROTATIONAL_PHASES, endpoint=False)
window_size = 4/24 # how many hours per night
sub_offsets = np.linspace(-window_size/2, window_size/2, HOUR_ANGLES)
times = jnp.array([time + sub_offsets for time in times]).flatten()

t_lc = jnp.linspace(0.,PERIOD,2000, endpoint=False)

radius = 1.47/2.
star_interferometry = Harmonix(star, radius)

vis_true = visibilities(star_interferometry, jnp.array(u.T), jnp.array(v.T), times)
light_curve_true = vmap(partial(surface_light_curve, star_interferometry.surface, r=0., x=1., y=1., z=1.))(theta=star_interferometry.rotational_phase(t_lc))

fig1, ax_data = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
ax_data[0].plot(jnp.sqrt(u**2+v**2), vis_true.T[:,:,0], alpha=0.5,color='k', lw=0.5, rasterized=True)
ax_data[0].set_ylim(0,1.0)
ax_data[1].scatter(t_lc, light_curve_true, s=1, color='k', alpha=0.5)

plt.savefig(paths.figures / "data_cta_north.png", dpi=300)
snr_lc = 1e4
losses_all = []

snr_cta = 200


key = jr.PRNGKey(0)
vis_data = vis_true + jr.normal(key, vis_true.shape)/snr_cta
lc_data = light_curve_true + jr.normal(key, light_curve_true.shape)/snr_lc

# Now lets construct a loss function
l_max = 15
n_max = l_max**2 + 2 * l_max + 1
y_star = np.zeros(n_max)
y_star[0] = 1.0  # Set the zeroth coefficient to 1 for a constant term
#y_star = np.load(paths.data / "SPOT_map_highres.npy")
#y_star[1:] += jr.normal(key, (n_max-1,))*0.1
y_star = jnp.array(y_star, dtype=jnp.float64)  # Ensure y_star is a JAX array
y = Ylm.from_dense(y_star)
star = Surface(y=y, inc=jnp.radians(60.), obl=0, period=PERIOD)
model = Harmonix(star, radius)
model_init = model

# Evaluate loss function once Compile to XLA
print("Evaluating loss function once to compile")
loss, grads = loss_fn(model, vis_data, u, v, times, t_lc, lc_data)
print("Loss function evaluated.")

opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-4)
)

# Get optax objcets
optimiser, state = zdx.get_optimiser(model, opt_params, opt)

losses = []

print("Starting optimization for CTA")
with tqdm(range(500)) as pbar:
    for i in pbar:
        loss, grads = loss_fn(model, vis_data, u, v, times, t_lc, lc_data)
        step, state = optimiser.update(grads, state, params=model)
        model = zdx.apply_updates(model, step)
        pbar.set_postfix(loss=f"{loss:.4f}")
        losses.append(loss)
print("Optimization completed for CTA")

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

fig, ax = plt.subplots(subplot_kw={'projection': 'mollweide'}, figsize=(10, 5))

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
ax.set_title(f"CTA North", fontsize=14)
plt.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
losses_all.append(losses)

plt.savefig(paths.figures / 'cta_optimization.pdf', bbox_inches='tight', dpi=300)
