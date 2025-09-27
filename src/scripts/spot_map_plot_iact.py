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

import pandas as pd

plt.rcParams.update({'font.size': 12})

load = Loader(paths.data)


PERIOD = 5.1  # Rotation period of the star in days
ROTATIONAL_PHASES = 8  # Number of rotational phases to consider

HOUR_ANGLES = 10

WAVS = 1

def plot_veritas(ax):
    """Plot VERITAS telescope positions on the given axis."""
    veritas_tels = np.array([[140, -10], [50, -50], [40, 60], [-30, 10]])
    station_x = veritas_tels[:, 0]
    station_y = veritas_tels[:, 1]

    ax.scatter(station_x, station_y, c="blue", s=50)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("VERITAS", fontsize=14, weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, color="k", alpha=0.5)

def plot_cta(ax, csv_path):
    """Plot CTA North telescope positions on the given axis."""
    df = pd.read_csv(csv_path)

    # If Type column exists, use it. Otherwise assume first 4 are LSTs.
    if "Type" in df.columns:
        lsts = df[df["Type"] == "LST"]
        msts = df[df["Type"] == "MST"]
    else:
        lsts = df.iloc[:4]
        msts = df.iloc[4:]

    # Plot MSTs
    ax.scatter(msts["East [m]"], msts["North [m]"],
               c="black", marker="s", s=50, label="MSTs")
    # Plot LSTs
    ax.scatter(lsts["East [m]"], lsts["North [m]"],
               facecolors="none", edgecolors="red", marker="o", s=80, label="LSTs")

    # Formatting
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("CTA North", fontsize=14, weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-600, 600)
    ax.set_ylim(-600, 600)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Outer circle ~600 m
    circle = plt.Circle((0, 0), 600, color="gray", linestyle="--", fill=False, alpha=0.5)
    ax.add_artist(circle)

    # Telescope counts
    ax.text(-580, -550,
            f"#LSTs: {len(lsts)}  #MSTs: {len(msts)}  #SSTs: 0",
            fontsize=9)

def plot_veritas_and_cta(csv_path):
    """Plot VERITAS (left) and CTA North (right) side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_veritas(axes[0])
    plot_cta(axes[1], csv_path)

    plt.tight_layout()
    plt.savefig(paths.figures / "iact_uv_coverage.pdf", bbox_inches="tight", dpi=300)

plot_veritas_and_cta(paths.data / "CTA_North_positions_omega.csv")


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


# ---------------------
# Set up the plot layout
fig = plt.figure(figsize=(11, 14))
fig.subplots_adjust(hspace=0.2, wspace=0.0)

# Mollweide map: use full width (colspan=8)
ax = [
    plt.subplot2grid((45, 8), (0, 0), rowspan=8, colspan=2,
                     projection='mollweide'),  # add projection here
    plt.subplot2grid((45, 8), (0, 3), rowspan=8, colspan=2),  # add projection here

    plt.subplot2grid((45, 8), (0, 6), rowspan=8, colspan=2)  # add projection here

]

# One row of 8 small orthographic views
ax_ortho = [
    plt.subplot2grid((45, 8), (10, n), rowspan=4, colspan=1) for n in range(8)
]

# Better: explicitly share with the first axis
ax_vis = [
    plt.subplot2grid((45, 8), (14, 0), rowspan=12, colspan=4),
]
ax_vis.append(
    plt.subplot2grid((45, 8), (14, 4), rowspan=12, colspan=4,
                     sharey=ax_vis[0])
)

ax_lc = [plt.subplot2grid((45, 8), (28, 0), rowspan=8, colspan=8)]


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
times = jnp.linspace(0, PERIOD, 8)  # Example times for the orthographic views
for n in range(8):
    show_surface(star, ax=ax_ortho[n], cmap='plasma', theta=star.rotational_phase(times[n]))

print("Plotting VERITAS")

veritas_tels = np.array([[140,-10],[50,-50],[40,60],[-30,10]])
station_x = veritas_tels[:,0]
station_y = veritas_tels[:,1]

baseline_inds, baselines = makebaselines(np.vstack([station_x, station_y]).T)

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
baseline_lengths = np.sqrt(u**2 + v**2)          # shape (Ntime, Nbaselines)
baseline_max = baseline_lengths.max(axis=0)      # one value per baseline

# Normalize to [0,1] for colormap
norm = baseline_max / baseline_max.max()

# Map to plasma colormap
cmap = plt.cm.winter
colors = cmap(norm)   # shape (Nbaselines, 4)
for i in range(u.shape[1]):
    ax[1].scatter(u[:,i],v[:,i],color=colors[i],s=2.)
    ax[1].scatter(-u[:,i],-v[:,i],color=colors[i],s=2.)
ax[1].set_xlabel("U (baseline/$\lambda$)")
ax[1].set_ylabel("V (baseline/$\lambda$)")

window_size = 4/24 # how many hours per night
sub_offsets = np.linspace(-window_size/2, window_size/2, HOUR_ANGLES)

radius = 1.47/2.
star_interferometry = Harmonix(star, radius)
vis_data = jnp.array([visibilities(star_interferometry, jnp.array(u.T), jnp.array(v.T), time + sub_offsets) for time in times])

print("Visibility data shape: " + str(vis_data.shape))
for n in range(ROTATIONAL_PHASES):
    for i in range(u.shape[1]):
        ax_vis[0].plot(jnp.sqrt(u[:,i]**2+v[:,i]**2), vis_data[n,:,i,:].T, alpha=1.0,color=colors[i], lw=0.5, rasterized=True)
    

ax_vis[0].set_xlabel(r'Spatial Frequency ($\lambda$)')
ax_vis[0].set_xlim(left=0)
ax_vis[0].set_ylim(bottom=0, top=1.0)
ax_vis[0].set_ylabel('Visibility Amplitude')

print("Plotting CTA North")

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


print("Plotting uv coverage...")

ax[2].set_aspect("equal", adjustable="datalim")

print("u shape: " + str(u.shape))
print("v shape: " + str(v.shape))
#wavs = wav.repeat(HOUR_ANGLES,axis=0).repeat(u.shape[1], axis=0)
baseline_lengths = np.sqrt(u**2 + v**2)          # shape (Ntime, Nbaselines)
baseline_max = baseline_lengths.max(axis=0)      # one value per baseline

# Normalize to [0,1] for colormap
norm = baseline_max / baseline_max.max()

# Map to winter colormap
cmap = plt.cm.winter
colors = cmap(norm)   # shape (Nbaselines, 4)
for i in range(u.shape[1]):
    ax[2].scatter(u[:,i],v[:,i],color=colors[i],s=2.)
    ax[2].scatter(-u[:,i],-v[:,i],color=colors[i],s=2.)
#ax[2].set_xlabel("U (baseline/$\lambda$)")
ax[2].set_ylabel("V (baseline/$\lambda$)")

window_size = 4/24 # how many hours per night
sub_offsets = np.linspace(-window_size/2, window_size/2, HOUR_ANGLES)

radius = 1.47/2.
star_interferometry = Harmonix(star, radius)
vis_data = jnp.array([visibilities(star_interferometry, jnp.array(u.T), jnp.array(v.T), time + sub_offsets) for time in times])

print("Visibility data shape: " + str(vis_data.shape))
for n in range(ROTATIONAL_PHASES):
    for i in range(u.shape[1]):
        ax_vis[1].plot(jnp.sqrt(u[:,i]**2+v[:,i]**2), vis_data[n,:,i,:].T, alpha=1.0,color=colors[i], lw=0.5, rasterized=True)
    

ax_vis[1].set_xlabel(r'Spatial Frequency ($\lambda$)')
ax_vis[1].set_xlim(left=0)
# Hide y-axis ticks and labels on the right panel
ax_vis[1].tick_params(labelleft=False)
ax_vis[1].set_ylabel("")  # also remove the axis label if you add one

@zdx.filter_jit
def lc_func(model, t):
    theta = model.rotational_phase(t)
    y = Ylm.from_dense(jnp.concatenate([jnp.array([1.0]), model.data]))
    star = Surface(y=y, inc=model.surface.inc, obl=model.surface.obl, period=model.surface.period, u=model.surface.u)
    light_curve = vmap(partial(surface_light_curve, star, r=0., x=1., y=1., z=1.))(theta=theta)
    return light_curve

t_lc = jnp.linspace(0.,PERIOD,2000, endpoint=False)
light_curve = vmap(partial(surface_light_curve, star_interferometry.surface, r=0., x=1., y=1., z=1.))(theta=star_interferometry.rotational_phase(t_lc))

ax_lc[0].scatter(t_lc, light_curve, s=1, color='k', alpha=0.5)
ax_lc[0].set_xlabel("Time [days]")
ax_lc[0].set_ylabel("Normalized Flux")
ax_lc[0].set_xlim(0, PERIOD)
plt.savefig(paths.figures / 'spot_map_iact.pdf', bbox_inches="tight", dpi=300)