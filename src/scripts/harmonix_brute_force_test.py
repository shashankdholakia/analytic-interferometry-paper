from jax import config

config.update("jax_enable_x64", True)

import jax
from jax import jit
import jax.numpy as jnp
from jaxoplanet.starry import Ylm, Surface

from harmonix.harmonix import Harmonix
from harmonix.solution import solution_vector, transform_to_zernike
import numpy as np
from scipy.special import j1

import matplotlib.pyplot as plt

import paths

u, v = np.linspace(-500,500,64), np.linspace(-500,500,64)
wavel = 2e-6 # m

uu, vv = np.meshgrid(np.linspace(-500,500,64),np.linspace(-500,500,64))
uvgrid = np.vstack((uu.flatten(),vv.flatten())).T

mas2rad = jnp.pi / 180.0 / 3600.0/ 1000.0
radius = 1.0 # mas

def airy(w, lam, diam):
    '''Airy function for a circular aperture, evaluated on baselines uv (m) with diameter diam (mas) at wavelength lam (m)'''
    
    r = w/lam

    d = diam*mas2rad

    return 2 * j1(jnp.pi * r * d) / (jnp.pi * r * d)

# Define the spherical harmonic map
ylm = Ylm.from_dense(jnp.array([1.0]))
star = Surface(y=ylm, inc=0., obl=0, period=1.0)
# Time doesn't matter for a uniform map
t = 0.0
cvis = Harmonix(star, radius).model(uvgrid[:,0]/wavel, uvgrid[:,1]/wavel, t)

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
wgrid = np.sqrt(uvgrid[:,0]**2 + uvgrid[:,1]**2)
ft_anal = airy(np.sort(wgrid), wavel,2*radius)
v2_anal, phase_anal = jnp.abs(ft_anal)**2, jnp.angle(ft_anal)
ax1.plot(np.sort(wgrid),v2_anal, 'b-', label='Analytical Airy function')

v2, phase = jnp.abs(cvis)**2, jnp.angle(cvis)
ax1.plot(wgrid,v2,'k.', label='Harmonix Model')
ax1.set_ylabel(r"$V^2$")
plt.xlabel(r"Baseline")
ax1.legend()
inds = np.argsort(wgrid)
ax2.plot(wgrid,v2[inds]-v2_anal, 'k.', label='Residuals')
ax2.legend()
ax1.set_ylabel(r"Residuals")
ax1.set_xlabel(r"Baseline")

from jaxoplanet.starry.utils import ortho_grid



@jit
def compute_DFTM1(x,y,uv,wavel):
    '''Compute a direct Fourier transform matrix, from coordinates x and y (milliarcsec) to uv (metres) at a given wavelength wavel.'''

    # Convert to radians
    x = x * jnp.pi / 180.0 / 3600.0/ 1000.0
    y = y * jnp.pi / 180.0 / 3600.0/ 1000.0

    # get uv in nondimensional units
    uv = uv / wavel

    # Compute the matrix
    dftm = jnp.exp(-2j* jnp.pi* (jnp.outer(uv[:,0],x)+jnp.outer(uv[:,1],y)))

    return dftm

@jit
def apply_DFTM1(image,dftm):
    '''Apply a direct Fourier transform matrix to an image.'''
    image /= image.sum()
    return jnp.dot(dftm,image.ravel())

# Define the spherical harmonic map
np.random.seed(35)  # For reproducibility
"""
coeffs = jnp.array([1.00,  0.22,  0.19,  0.11,  0.11,  0.07,  -0.11, 0.00,  -0.05, 
     0.12,  0.16,  -0.05, 0.06,  0.12,  0.05,  -0.10, 0.04,  -0.02, 
     0.01,  0.10,  0.08,  0.15,  0.13,  -0.11, -0.07, -0.14, 0.06, 
     -0.19, -0.02, 0.07,  -0.02, 0.07,  -0.01, -0.07, 0.04,  0.00])
"""
coeffs = np.load(paths.data / "SPOT_map_lowres.npy")
                    
ylm = Ylm.from_dense(coeffs)
star = Surface(y=ylm, inc=jnp.radians(90.), obl=0.0, period=1.0)

res = 400
x, y = jnp.meshgrid(jnp.linspace(-1, 1, res), jnp.linspace(-1, 1, res))
image = star.render(res=res,theta=jnp.radians(0.))
image = jnp.nan_to_num(image, nan=0.0)
plt.imshow(image, origin='lower', cmap='gray', extent=(-1, 1, -1, 1))
plt.colorbar()

dftm = compute_DFTM1(x, y, uvgrid, wavel)
cvis_dftm = apply_DFTM1(image, dftm)

mas2rad = jnp.pi / 180.0 / 3600.0/ 1000.0
radius = 1.0 # mas

cvis = Harmonix(star, radius).model(uvgrid[:,0]/wavel, uvgrid[:,1]/wavel, 0.0)
# --- 1. Increase Font Sizes ---
# Adjust these values as needed
plt.rcParams.update({
    'font.size': 14,          # General font size
    'axes.titlesize': 16,     # Title size
    'axes.labelsize': 14,     # X and Y label size
    'xtick.labelsize': 12,    # Tick size
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# --- 2. Calculate Data (Preserving your logic) ---
v2_dftm, phase_dftm = jnp.abs(cvis_dftm)**2, jnp.angle(cvis_dftm)
v2, phase = jnp.abs(cvis)**2, jnp.angle(cvis)

# --- 3. Create 2x2 Plot ---
# width=18, height=8 to accommodate two side-by-side panels
fig, axes = plt.subplots(2, 2, figsize=(18, 8), sharex='col', 
                         gridspec_kw={'height_ratios': [3, 1], 'wspace': 0.2, 'hspace': 0.05})

# Unpack axes for easier reference
# axes[row, col]
ax_vis       = axes[0, 0]  # Top Left
ax_vis_res   = axes[1, 0]  # Bottom Left
ax_phase     = axes[0, 1]  # Top Right
ax_phase_res = axes[1, 1]  # Bottom Right

# ==========================================
# LEFT COLUMN: Visibility Squared
# ==========================================

# Main Plot
ax_vis.plot(wgrid, v2_dftm, 'ro', label='DFTM Visibility', zorder=3, alpha=1.0, ms=4.)
ax_vis.plot(wgrid, v2, 'k.', label='Harmonix Model', alpha=1.0, zorder=4)
ax_vis.set_ylabel(r"$V^2$")
ax_vis.legend(loc='upper right')

# Residuals (Difference)
ax_vis_res.plot(wgrid, v2 - v2_dftm, 'k.', label='Residuals')
ax_vis_res.set_ylabel(r"Residual")
ax_vis_res.set_xlabel(r"Baseline")
ax_vis_res.legend(loc='lower right')

# ==========================================
# RIGHT COLUMN: Phase
# ==========================================

# Main Plot
ax_phase.plot(wgrid, phase_dftm, 'ro', label='DFTM Visibility', ms=4)
ax_phase.plot(wgrid, phase, 'k.', label='Harmonix Model')
ax_phase.set_ylabel(r"Phase (radians)")
ax_phase.legend(loc='lower right')

# Residuals (Ratio, based on your original code)
# Note: Original code used division (phase/phase_dftm). 
# If you meant difference (phase - phase_dftm), change the operator below.
ax_phase_res.plot(wgrid, phase - phase_dftm, 'k.', label='Residuals')
ax_phase_res.set_ylabel(r"Residual")
ax_phase_res.set_xlabel(r"Baseline")
ax_phase_res.set_ylim(-0.05, 0.05)
ax_phase_res.legend(loc='lower right')

# --- 4. Save ---
plt.savefig(paths.figures / "harmonix_vs_dftm_combined.png", bbox_inches="tight", dpi=300)