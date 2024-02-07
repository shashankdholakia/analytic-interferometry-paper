import numpy as np
import matplotlib.pyplot as plt
from harmonix.solution import j_to_nm, nm_to_j, transform_to_zernike, zernike_FT, jmax, A, CHSH_FT

from matplotlib import colors
import paths

def static(lmax=5, res=300):
    """Plot a static PDF figure."""
    # Set up the plot
    fig, ax = plt.subplots(lmax + 1, 2 * lmax + 1, figsize=(9, 4))
    fig.subplots_adjust(hspace=0, wspace=0)
    
    # Find the min and max of all colors for use in setting the color scale.
    vmin = 0
    vmax = 1
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for axis in ax.flatten():
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.set_aspect('equal')
    for l in range(lmax + 1):
        ax[l, 0].set_ylabel(r"$l = %d$" % l,
                            rotation='horizontal',
                            labelpad=30, y=0.38,
                            fontsize=8)
    for j, m in enumerate(range(-lmax, lmax + 1)):
        if m < 0:
            ax[-1, j].set_xlabel(r"$m {=} \mathrm{-}%d$" % -m,
                                 labelpad=2, fontsize=4)
        else:
            ax[-1, j].set_xlabel(r"$m = %d$" % m, labelpad=2, fontsize=4)

    # Fiducial UV plane
    r = np.linspace(0, 15, res)
    theta = np.linspace(0, 2*np.pi, res)
    rho, phi = np.meshgrid(r, theta)
    
    X, Y = rho*np.cos(phi), rho*np.sin(phi)

    images = []
    # Loop over the orders and degrees
    for i, l in enumerate(range(lmax + 1)):
        for j, m in enumerate(range(-l, l + 1)):

            # Offset the index for centered plotting
            j += lmax - l

            # Compute the spherical harmonic
            # with no rotation
            # Plot the spherical harmonic
            ft = np.zeros_like(rho, dtype='complex128')
            if (l+m)%2==0:
                y_hsh = np.zeros(jmax(5)+1)
                y_hsh[nm_to_j(l,m)] = 1
                zs = transform_to_zernike(y_hsh)
                for k in range(len(zs)):
                    n_,m_ = j_to_nm(k)
                    ft += zs[k]*zernike_FT(n_,m_)(rho,phi)
            else:
                ft += CHSH_FT(l,m)(rho,phi)
            img = ax[i, j].pcolormesh(X, Y, np.abs(ft), cmap='plasma')
            images.append(img)
            img.set_norm(norm)
            ax[i, j].set_xlim(-16, 16)
            ax[i, j].set_ylim(-16,16)

    # Save!
    #fig.suptitle("Vis squared for each spherical harmonic")
    fig.colorbar(images[0], ax=ax[0,-1], orientation='horizontal', fraction=1)
    fig.savefig(paths.figures / "ylms_nullspace_amp.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    
static(lmax=5)