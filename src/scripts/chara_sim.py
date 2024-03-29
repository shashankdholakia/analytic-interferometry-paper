import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import zodiax as zdx

import numpy as np
from harmonix.harmonix import Harmonix, visibilities, closure_phases
from harmonix.utils import maketriples_all, makebaselines
from jaxoplanet.experimental.starry import Map, Ylm, show_map
from jaxoplanet.experimental.starry.light_curves import map_light_curve


import matplotlib.pyplot as plt
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

def loglike_visibility(model, data, noise, radius, u, v, t):
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
    mas_to_rad = 1/(1000*60*60*180)*jnp.pi**2
    vis = visibilities(model, radius*mas_to_rad*u, radius*mas_to_rad*v, t)
    return -0.5 * jnp.sum((vis - data) ** 2 / noise ** 2)

def loglike_cp(model, data, noise, radius, u, v, t, index_cps1, index_cps2, index_cps3):
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
    mas_to_rad = 1/(1000*60*60*180)*jnp.pi**2
    cp = closure_phases(model, radius*mas_to_rad*u, radius*mas_to_rad*v, t, index_cps1, index_cps2, index_cps3)
    return -0.5 * jnp.sum((cp - data) ** 2 / noise ** 2)

def loglike_photometry(model, data, noise, t):
    theta = model.rotational_phase(t)
    y = Ylm.from_dense(model.data)
    star = Map(y=y, inc=star_interferometry.map.inc, obl=star_interferometry.map.obl, period=star_interferometry.map.period, u=star_interferometry.map.u)
    light_curve = vmap(partial(map_light_curve, star, r=0., xo=1., yo=1., zo=1.))(theta=theta)
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
plt.xlabel("x")
plt.ylabel("y")
plt.title("CHARA telescope positions")
plt.scatter(station_x, station_y)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(paths.figures / "chara_tels.pdf", dpi=300)

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

print("Plotting uv coverage...")
fig1, ax = plt.subplots()
ax.set_aspect("equal", adjustable="datalim")
u = np.concatenate(uv_by_wav[:,:,0],axis=0)
v = np.concatenate(uv_by_wav[:,:,1],axis=0)
wavs = wav.repeat(HOUR_ANGLES,axis=0).repeat(u.shape[1], axis=0)
ax.scatter(u,v,c=wavs,cmap='rainbow',s=1.);
ax.set_xlabel("U (lambdas)")
ax.set_ylabel("V (lambdas)")
fig1.savefig(paths.figures / 'alioth_uv_coverage.pdf', dpi=300)

print("Loading star map...")
y_star = np.load(paths.data / "spot_map.npy")
y = Ylm.from_dense(y_star)
star = Map(y=y, inc=jnp.radians(90.), obl=0, period=1.0, u=[0.1, 0.1])
star_interferometry = Harmonix(star)
radius = 1.47

mas_to_rad = 1/(1000*60*60*180)*jnp.pi**2
t = jnp.linspace(0,1,ROTATIONAL_PHASES, endpoint=False)
noise = 0.01
vis_data = visibilities(star_interferometry, radius*mas_to_rad*jnp.array(u.T), radius*mas_to_rad*jnp.array(v.T),t)
vis_data += jax.random.normal(jax.random.PRNGKey(0), vis_data.shape)*noise
cp_data = closure_phases(star_interferometry, radius*mas_to_rad*jnp.array(u.T), radius*mas_to_rad*jnp.array(v.T),t, cp_inds[0:10,0], cp_inds[0:10,1], cp_inds[0:10,2])
cp_data += jax.random.normal(jax.random.PRNGKey(0), cp_data.shape)*noise*360


print("Plotting visibilities and closure phases of the simulated CHARA observations...")
for i in np.arange(ROTATIONAL_PHASES):
    fig = plt.figure()
    plt.scatter(jnp.sqrt((radius*mas_to_rad*jnp.array(u.T))**2+(radius*mas_to_rad*jnp.array(v.T))**2), vis_data[i,:,:], s=1, c='k');
    plt.ylim([0,1])
    plt.xlabel("spatial frequency")
    plt.ylabel("visibility amplitude")
    plt.savefig(paths.figures / f'vis_data_{i}.pdf', dpi=300)
    
    cp_x_axis = jnp.max(
    jnp.array([jnp.sqrt((radius*mas_to_rad*jnp.array(u.T))**2+(radius*mas_to_rad*jnp.array(v.T))**2)[cp_inds[0:10,0]],
    jnp.sqrt((radius*mas_to_rad*jnp.array(u.T))**2+(radius*mas_to_rad*jnp.array(v.T))**2)[cp_inds[0:10,1]],
    jnp.sqrt((radius*mas_to_rad*jnp.array(u.T))**2+(radius*mas_to_rad*jnp.array(v.T))**2)[cp_inds[0:10,2]]]), axis=0)
    plt.scatter(cp_x_axis, cp_data[i,:,:], s=1, c='k');
    plt.xlabel("spatial frequency")
    plt.ylabel("closure phase")


opt_params = ['data']
print("Creating the Fisher information matrices...")
fim_vis = -zdx.fisher_matrix(star_interferometry, opt_params,loglike_visibility, 
                            data=vis_data, radius=radius, 
                            u=u.T, v=v.T,t=t, noise=noise)
fim_cp = -zdx.fisher_matrix(star_interferometry, opt_params,loglike_cp, data=cp_data, radius=radius, u=u.T, v=v.T,t=t, noise=noise*360.,
                           index_cps1=cp_inds[0:10,0], index_cps2=cp_inds[0:10,1], index_cps3=cp_inds[0:10,2])

lm_to_n = lambda l,m : l**2+l+m
l_max = lambda y: int(jnp.floor(jnp.sqrt(len(y)-1)))
lmax = l_max(y_star)

def rearrange_m_inds(l_max):
    inds = []
    for m in range(-l_max,l_max+1):
        for l in range(abs(m), l_max+1):
            inds.append(lm_to_n(l,m))
    return jnp.array(inds)

print("Plotting the Fisher information matrices...")
fig = plt.figure()
plt.imshow(fim_vis,vmin=-jnp.max(fim_vis),vmax=jnp.max(fim_vis),  cmap='RdBu_r')
plt.colorbar()
tick_labels = jnp.arange(0,lmax+1)
tick_positions = [nmax(i)-1 for i in tick_labels]
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels)
plt.grid(True,linestyle=':',color='black',alpha=1)
plt.savefig(paths.figures / f'FIM_vis.pdf')

fig = plt.figure()
plt.imshow(fim_cp,vmin=-jnp.max(fim_cp),vmax=jnp.max(fim_cp),  cmap='RdBu_r')
plt.colorbar()
tick_labels = jnp.arange(0,lmax+1)
tick_positions = [nmax(i)-1 for i in tick_labels]
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels)
plt.grid(True,linestyle=':',color='black',alpha=1)
plt.savefig(paths.figures / f'FIM_cp.pdf')

fig = plt.figure()
plt.imshow(fim_cp+fim_vis,vmin=-jnp.max(fim_cp+fim_vis),vmax=jnp.max(fim_cp+fim_vis),  cmap='RdBu_r')
plt.colorbar()
tick_labels = jnp.arange(0,lmax+1)
tick_positions = [nmax(i)-1 for i in tick_labels]
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels)
plt.grid(True,linestyle=':',color='black',alpha=1)
plt.savefig(paths.figures / f'FIM_vis_cp.pdf')

print("Plotting the covariance matrices...")
fig = plt.figure()
cov_vis = -jnp.linalg.inv(-(fim_vis)[1:,1:])
plt.imshow(cov_vis, cmap='RdBu_r',vmin=-jnp.max(cov_vis),vmax=jnp.max(cov_vis))
plt.colorbar()
tick_labels = jnp.arange(1,lmax+1)
tick_positions = [nmax(i)-2 for i in tick_labels]
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels)
plt.grid(True,linestyle=':',color='black',alpha=1)
plt.savefig(paths.figures / f'cov_vis.pdf')

fig = plt.figure()
cov_cp = -jnp.linalg.inv(-(fim_cp)[1:,1:])
plt.imshow(cov_cp, cmap='RdBu_r',vmin=-jnp.max(cov_cp),vmax=jnp.max(cov_cp))
plt.colorbar()
tick_labels = jnp.arange(1,lmax+1)
tick_positions = [nmax(i)-2 for i in tick_labels]
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels)
plt.grid(True,linestyle=':',color='black',alpha=1)
plt.savefig(paths.figures / f'cov_cp.pdf')

fig = plt.figure()
cov = -jnp.linalg.inv(-(fim_cp+fim_vis)[1:,1:])
plt.imshow(cov, cmap='RdBu_r',vmin=-jnp.max(cov),vmax=jnp.max(cov))
plt.colorbar()
tick_labels = jnp.arange(1,lmax+1)
tick_positions = [nmax(i)-2 for i in tick_labels]
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels)
plt.grid(True,linestyle=':',color='black',alpha=1)
plt.savefig(paths.figures / f'cov_vis_cp.pdf')


fig = plt.figure()
t_lc = jnp.linspace(0,10,1000, endpoint=False)
light_curve_data = vmap(partial(map_light_curve, star_interferometry.map, r=0., xo=1., yo=1., zo=1.))(theta=star_interferometry.rotational_phase(t_lc))
lc_noise = 1e-5
light_curve_data+= jax.random.normal(jax.random.PRNGKey(0), light_curve_data.shape)*lc_noise
plt.plot(t_lc, light_curve_data)

print("Creating the Fisher information matrix for the light curve...")
fim_lc = -zdx.fisher_matrix(star_interferometry, opt_params,loglike_photometry, data=light_curve_data, noise=lc_noise, t=t_lc)

print("Plotting the Fisher info for the light curve...")
fig = plt.figure()
plt.imshow(fim_lc,vmin=-jnp.max(fim_lc),vmax=jnp.max(fim_lc),  cmap='RdBu_r')
plt.colorbar()
tick_labels = jnp.arange(0,lmax+1)
tick_positions = [nmax(i)-1 for i in tick_labels]
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels)
plt.grid(True,linestyle=':',color='black',alpha=1)
plt.savefig(paths.figures / f'FIM_lc.pdf')

total_fim = fim_cp+fim_vis+fim_lc
fig = plt.figure()
plt.imshow(total_fim,vmin=-jnp.max(total_fim),vmax=jnp.max(total_fim),  cmap='RdBu_r')
plt.colorbar()
tick_labels = jnp.arange(0,lmax+1)
tick_positions = [nmax(i)-1 for i in tick_labels]
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels)
plt.grid(True,linestyle=':',color='black',alpha=1)
plt.savefig(paths.figures / f'FIM_total.pdf')

fig = plt.figure()
total_cov = jnp.linalg.inv(total_fim[1:,1:])
plt.imshow(total_cov, cmap='RdBu_r',vmin=-jnp.max(total_cov),vmax=jnp.max(total_cov))
plt.colorbar()
tick_labels = jnp.arange(1,lmax+1)
tick_positions = [nmax(i)-2 for i in tick_labels]
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels)
plt.grid(True,linestyle=':',color='black',alpha=1)
plt.savefig(paths.figures / f'cov_vis_cp_lc.pdf')