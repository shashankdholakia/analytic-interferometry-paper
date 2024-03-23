import numpy as np
import matplotlib.pyplot as plt
from harmonix.solution import j_to_nm, nm_to_j, transform_to_zernike, zernike_FT, jmax, A, CHSH_FT
from jaxoplanet.experimental.starry import Map, Ylm, show_map
import matplotlib.font_manager

from matplotlib import colors
import paths

nmax = lambda l_max: l_max**2 + 2 * l_max + 1

lm_to_n = lambda l,m : l**2+l+m

# Style
plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["font.size"] = 14
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["font.cursive"] = ["Liberation Sans"]
try:
    plt.rcParams["mathtext.fallback"] = "cm"
except KeyError:
    plt.rcParams["mathtext.fallback_to_cm"] = True

def ylms(lmax=5, res=300, hsh=False):
    """Plot a static PDF figure."""
    # Set up the plot
    fig, ax = plt.subplots(lmax + 1, 2 * lmax + 1, figsize=(9, 6))
    fig.subplots_adjust(hspace=0)
    for axis in ax.flatten():
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.set_rasterization_zorder(-1)
    for l in range(lmax + 1):
        ax[l, 0].set_ylabel(
            "l=%d" % l,
            rotation="horizontal",
            labelpad=20,
            y=0.38,
            fontsize=10,
            alpha=0.5,
        )
    for j, m in enumerate(range(-lmax, lmax + 1)):
        ax[-1, j].set_xlabel("m=%d" % m, labelpad=10, fontsize=10, alpha=0.5)

    # Plot it
    #x = np.linspace(-1, 1, res)
    #y = np.linspace(-1, 1, res)
    #X, Y = np.meshgrid(x, y)
   

    # Loop over the orders and degrees
    for i, l in enumerate(range(lmax + 1)):
        for j, m in enumerate(range(-l, l + 1)):

            # Offset the index for centered plotting
            j += lmax - l

            # Compute the spherical harmonic
            # with no rotation
            y = np.zeros(nmax(lmax))
            y[0] = 1
            y[lm_to_n(l,m)] = 1
            map = Ylm.from_dense(y)
            star_map = Map(y=map)
            flux = star_map.render(res=res)
            # Plot the spherical harmonic
            if not hsh:
                ax[i, j].imshow(flux, cmap='plasma',
                            interpolation="none", origin="lower",
                            extent=(-1, 1, -1, 1))
            else:
                if (l+m)%2==0:
                    ax[i, j].imshow(flux, cmap='Reds_r',
                            interpolation="none", origin="lower",
                            extent=(-1, 1, -1, 1))
                else:
                    ax[i, j].imshow(flux, cmap='Blues_r',
                            interpolation="none", origin="lower",
                            extent=(-1, 1, -1, 1))
            ax[i, j].set_xlim(-1.1, 1.1)
            ax[i, j].set_ylim(-1.1, 1.1)

    # Save!
    append = '' if (not hsh) else '_hsh_colors'
    fig.savefig(paths.figures / f"ylms{append}.png", bbox_inches="tight", dpi=300)
    #default backend somehow doesn't plot l,m=0 for pdf specifically
    fig.savefig(paths.figures / f"ylms{append}.pdf", bbox_inches="tight", backend='pgf', dpi=300)
    plt.close()

ylms()
ylms(hsh=True)
ylms(hsh=False)

"""
if __name__ == "__main__":
    ylms()
    ylms(hsh=True)
    ylms(hsh=False)
"""