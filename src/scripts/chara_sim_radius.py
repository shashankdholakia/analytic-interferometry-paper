import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import zodiax as zdx

import numpy as np
from harmonix.harmonix import Harmonix, visibilities, closure_phases
from harmonix.utils import maketriples_all, makebaselines
from jaxoplanet.starry import Surface, Ylm, show_surface
from jaxoplanet.starry.light_curves import surface_light_curve


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import paths
from functools import partial

from skyfield.api import load
from skyfield.api import Star
from skyfield.api import Loader

from skyfield.data import hipparcos
from skyfield.api import N,S,E,W, wgs84

load = Loader(paths.data)

ROTATIONAL_PHASES = 6
UV_MAX = 8

HOUR_ANGLES = 5
#using SPICA lowres
WAVS = 50

def loglike_visibility(model, data, noise, u, v, t):
    """Log likelihood for visibility amplitude

    Args:
        model (Harmonix): Harmonix model
        radius (float): Radius of the star
        u (jnp.array): N_baselines x N_samples array of u coordinates
        v (jnp.array): N_baselines x N_samples array of v coordinates
        t (float): (temporary) float of time t

    Returns:
        jnp.array: N_samples x N_baselines array of visibility amplitudes
    """
    vis = visibilities(model, u, v, t)
    return -0.5 * jnp.sum((vis - data) ** 2 / noise ** 2)

def loglike_cp(model, data, noise, u, v, t, index_cps1, index_cps2, index_cps3):
    """Log likelihood for visibility amplitude

    Args:
        model (Harmonix): Harmonix model
        radius (float): Radius of the star
        u (jnp.array): N_baselines x N_samples array of u coordinates
        v (jnp.array): N_baselines x N_samples array of v coordinates
        t (float): (temporary) float of time t

    Returns:
        jnp.array: N_samples x N_baselines array of visibility amplitudes
    """
    cp = closure_phases(model, u, v, t, index_cps1, index_cps2, index_cps3)
    return -0.5 * jnp.sum((cp - data) ** 2 / noise ** 2)

def loglike_photometry(model, data, noise, t):
    theta = model.rotational_phase(t)
    y = Ylm.from_dense(model.data)
    star = Surface(y=y, inc=star_interferometry.surface.inc, obl=star_interferometry.surface.obl, period=star_interferometry.surface.period, u=star_interferometry.surface.u)
    light_curve = vmap(partial(surface_light_curve, star, r=0., x=0., y=0., z=0.))(theta=theta)
    return -0.5 * jnp.sum((light_curve - data) ** 2 / noise ** 2)

nmax = lambda l_max: l_max**2 + 2 * l_max + 1


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
baseline_inds, baselines = makebaselines(np.vstack([station_x, station_y]).T)


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
wavs = wav.repeat(HOUR_ANGLES,axis=0).repeat(u.shape[1], axis=0)

print("Loading star surface...")
y_star = np.load(paths.data / "SPOT_map_highres.npy")
y = Ylm.from_dense(y_star)
incs = [60.]
for inc in incs:
    star = Surface(y=y, inc=jnp.radians(inc), obl=0., period=1.0, u=jnp.array([0.1,0.1]))
    lm_to_n = lambda l,m : l**2+l+m
    l_max = lambda y: int(jnp.floor(jnp.sqrt(len(y)-1)))
    lmax = l_max(y_star)

    def rearrange_m_inds(l_max):
        inds = []
        for m in range(-l_max,l_max+1):
            for l in range(abs(m), l_max+1):
                inds.append(lm_to_n(l,m))
        return jnp.array(inds)

    radii = jnp.linspace(0.1, 2.0, 25)
    max_baselines = []


    det = np.zeros((radii.shape[0], lmax + 1))
    det_total = np.zeros((radii.shape[0], lmax + 1))

    for i, radius in enumerate(radii):
        mas_to_rad = 1/(1000*60*60*180)*jnp.pi**2
        star_interferometry = Harmonix(star, radius)
        max_baselines.append(jnp.max(jnp.sqrt((radius*mas_to_rad*jnp.array(u.T))**2+(radius*mas_to_rad*jnp.array(v.T))**2)))
        t = jnp.linspace(0,1,ROTATIONAL_PHASES, endpoint=False)
        noise = 0.01
        vis_data = visibilities(star_interferometry, jnp.array(u.T), jnp.array(v.T),t)
        #vis_data += jax.random.normal(jax.random.PRNGKey(1), vis_data.shape)*noise #don't have to actually add in noise for custom loglike
        cp_data = closure_phases(star_interferometry, jnp.array(u.T), jnp.array(v.T),t, cp_inds[0:10,0], cp_inds[0:10,1], cp_inds[0:10,2])
        #cp_data += jax.random.normal(jax.random.PRNGKey(1), cp_data.shape)*noise*360 #don't have to actually add in noise for custom loglike
        

        opt_params = ["data","u"]
        print(f"Creating the Fisher information matrices for a radius of {radius}...")

        fim_vis = -zdx.fisher_matrix(star_interferometry, opt_params,loglike_visibility, 
                                    data=vis_data, 
                                    u=u.T, v=v.T,t=t, noise=noise)
        fim_cp = -zdx.fisher_matrix(star_interferometry, opt_params,loglike_cp, data=cp_data, u=u.T, v=v.T,t=t, noise=noise*360.,
                                index_cps1=cp_inds[0:10,0], index_cps2=cp_inds[0:10,1], index_cps3=cp_inds[0:10,2])

        t_lc = jnp.linspace(0,10,1000, endpoint=False)
        light_curve_data = vmap(partial(surface_light_curve, star_interferometry.surface, r=0., x=0., y=0., z=0.))(theta=star_interferometry.rotational_phase(t_lc))
        lc_noise = 1e-4
        #light_curve_data+= jax.random.normal(jax.random.PRNGKey(1), light_curve_data.shape)*lc_noise #don't have to actually add in noise for custom loglike
        fim_lc = -zdx.fisher_matrix(star_interferometry, opt_params,loglike_photometry, data=light_curve_data, noise=lc_noise, t=t_lc)
        for l in range(lmax + 1):
            indices = [lm_to_n(l, m) for m in range(-l, l + 1)]
            #check this logic carefully
            block_vis = fim_vis[np.ix_(indices, indices)]
            block_cp = fim_cp[np.ix_(indices, indices)]
            
            _, det[i, l] = np.linalg.slogdet(block_vis + block_cp)
            block_lc = fim_lc[np.ix_(indices, indices)]
            _, det_total[i, l] = np.linalg.slogdet(block_vis + block_cp+block_lc)

    def ud(x):
        return 2*(jax.scipy.special.bessel_jn(x,v=1,n_iter=30)[1]/x)

    max_baselines = jnp.array(max_baselines)


    fig = plt.figure(figsize=(10,5))
    fig.subplots_adjust(right=0.85)
    ax = fig.add_subplot(111)
    x = jnp.linspace(0, jnp.max(max_baselines), 500)
    ax.plot(x/jnp.max(jnp.sqrt(u.T**2+v.T**2)*mas_to_rad), jnp.abs(ud(x)), label="Uniform disk (UD)", color='k')
    ax.set_ylabel('UD visibility amplitude at CHARA max baseline')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    for baseline, d_opt, d_total_opt in zip(max_baselines, det, det_total):
        print(baseline, d_opt, d_total_opt)
    l_vals = np.arange(lmax + 1)
    cmap = plt.cm.plasma
    boundaries = np.arange(-0.5, lmax + 1.5, 1)  # from -0.5 to lmax+0.5 for integer bins
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    xvals = max_baselines / jnp.max(jnp.sqrt(u.T**2 + v.T**2) * mas_to_rad)

    for l in range(lmax + 1):
        color = cmap(norm(l))  # Get color from colormap via norm
        ax2.plot(xvals, det[:, l], color=color, label=None)
        #ax2.plot(xvals, det_total[:, l], marker='^', color=color, label=None)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for older matplotlib versions
    cbar = plt.colorbar(sm, ax=ax2, ticks=l_vals, boundaries=boundaries, pad=0.1, fraction=0.046)
    cbar.set_label("Spherical harmonic degree $l$")
    cbar.ax.set_yticklabels([str(l) for l in l_vals])  # optional, for integer ticks
    cbar.ax.minorticks_off()
    ax2.set_ylabel('Fisher information $\ln{\mathrm{det}}$')
    ax.set_xlabel('Angular diameter (mas)')
    #ax2.set_yscale('log')
    #ax2.legend(loc='upper right')
    fig.savefig(paths.figures / f"chara_sim_radius_sqrt_inc{inc}_fim_u.pdf")
    
    fig = plt.figure(figsize=(10,5))
    fig.subplots_adjust(right=0.85)
    ax = fig.add_subplot(111)
    x = jnp.linspace(0, jnp.max(max_baselines), 500)
    ax.plot(x/jnp.max(jnp.sqrt(u.T**2+v.T**2)*mas_to_rad), jnp.abs(ud(x)), label="Uniform disk (UD)", color='k')
    ax.set_ylabel('UD visibility amplitude at CHARA max baseline')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    for baseline, d_opt, d_total_opt in zip(max_baselines, det, det_total):
        print(baseline, d_opt, d_total_opt)
    l_vals = np.arange(lmax + 1)
    cmap = plt.cm.plasma
    boundaries = np.arange(-0.5, lmax + 1.5, 1)  # from -0.5 to lmax+0.5 for integer bins
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    xvals = max_baselines / jnp.max(jnp.sqrt(u.T**2 + v.T**2) * mas_to_rad)

    for l in range(lmax + 1):
        color = cmap(norm(l))  # Get color from colormap via norm
        ax2.plot(xvals, det[:, l], marker='o', color=color, label=None)
        ax2.plot(xvals, det_total[:, l], marker='^', color=color, label=None)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for older matplotlib versions
    cbar = plt.colorbar(sm, ax=ax2, ticks=l_vals, boundaries=boundaries, pad=0.1, fraction=0.046)
    cbar.set_label("Spherical harmonic degree $l$")
    cbar.ax.set_yticklabels([str(l) for l in l_vals])  # optional, for integer ticks
    cbar.ax.minorticks_off()
    ax2.set_ylabel('Fisher information $\ln{\mathrm{det}}$')
    ax.set_xlabel('Angular diameter (mas)')
    #ax2.set_yscale('log')
    #ax2.legend(loc='upper right')
    fig.savefig(paths.figures / f"chara_sim_radius_sqrt_inc{inc}_fim_u_total.pdf")