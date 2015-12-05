"""
Generate nice plots providing additional information
"""
import sys, os

import numpy as np

import matplotlib.pylab as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from singularity_finder import compute_tau, compute_singularity_measure, compute_local_phase_field, compute_discrete_gradient, preprocess_data
from utils import save_data


def neural_spike():
    """ Plot various neural spikes
    """
    def do_plot(cell_evo):
        """ Plot [cAMP] for single cell over time
        """
        plt.plot(range(len(cell_evo)), cell_evo)

    camp, pacemaker = np.load(sys.argv[1])
    camp = np.rollaxis(camp, 0, 3)

    width, height, depth = camp.shape
    for j in range(width):
        for i in range(height):
            do_plot(camp[i, j])

    plt.title('cAMP concentration in single cell')
    plt.xlabel('t')
    plt.ylabel('cAMP concentration')

    plt.savefig('images/neural_spike.png', bbox_inches='tight', dpi=300)
    plt.show()

def lagged_phase_space():
    """ Plot phase space lagged by tau
    """
    def do_plot(cell_evo):
        x = []
        y = []
        for t in range(len(cell_evo) - tau):
            x.append(cell_evo[t + tau])
            y.append(cell_evo[t])

        plt.plot(x, y, 'o')

    camp, pacemaker = np.load(sys.argv[1])
    camp = np.rollaxis(camp, 0, 3)

    tau = compute_tau(camp)

    width, height, depth = camp.shape
    for j in range(width):
        for i in range(height):
            do_plot(camp[i, j])

    plt.title(r'Lagged phase space of neural spike ($\tau = %d$)' % tau)
    plt.xlabel(r'$x_{ij}(t - \tau)$')
    plt.ylabel(r'$x_{ij}(t)$')

    plt.savefig('images/lagged_phase_space.png', bbox_inches='tight', dpi=300)
    #plt.show()

def singularity_plot():
    """ Plot overview over singularity measure
    """
    cache_dir = 'cache'
    if not os.path.isdir(cache_dir):
        # preprocess input
        camp, pacemaker = np.load(sys.argv[1])
        camp = np.rollaxis(camp, 0, 3)
        camp = preprocess_data(camp)
        print(camp.shape)

        # compute data
        rolled_camp = np.rollaxis(camp, 2, 0)
        lphase = compute_local_phase_field(camp) # decreases last dim due to tau
        grads = compute_discrete_gradient(lphase)
        singularities = compute_singularity_measure(grads)

        # cache data
        save_data('%s/rolled_camp' % cache_dir, rolled_camp)
        save_data('%s/lphase' % cache_dir, lphase)
        save_data('%s/grads' % cache_dir, grads)
        save_data('%s/singularities' % cache_dir, singularities)
    else:
        print('Using cached data')
        rolled_camp = np.load('%s/rolled_camp.npy' % cache_dir)
        lphase = np.load('%s/lphase.npy' % cache_dir)
        grads = np.load('%s/grads.npy' % cache_dir)
        singularities = np.load('%s/singularities.npy' % cache_dir)

    # plot data
    pos_num = 4
    pos_range = range(0, lphase.shape[0], int(lphase.shape[0]/(pos_num)))[1:]
    fig, axarr = plt.subplots(len(pos_range), 4)
    fig.tight_layout()
    plt.suptitle('pipeline overview')

    def show(data, title, ax):
        ax.set_title(title)
        im = ax.imshow(
            data, interpolation='nearest', cmap=cm.gray)
        holder = make_axes_locatable(ax)
        cax = holder.append_axes('right', size='20%', pad=0.05)
        plt.colorbar(im, cax=cax, format='%.2f')

    for axrow, pos in enumerate(pos_range):
        show(rolled_camp[pos], 'cell overview', axarr[axrow][0])
        show(lphase[pos], 'local phase', axarr[axrow][1])
        show(grads[pos], 'gradient', axarr[axrow][2])
        show(singularities[pos], 'singularity measure', axarr[axrow][3])

    plt.savefig('images/singularity.png', bbox_inches='tight', dpi=300)
    #plt.show()

    plt.figure()
    avg_singularity = np.mean(singularities, axis=0)
    show(avg_singularity, 'averaged singularity measure', plt.gca())
    plt.savefig('images/averaged_singularity.png', bbox_inches='tight', dpi=300)
    #plt.show()


if __name__ == '__main__':
    dname = 'images'
    if not os.path.isdir(dname):
        os.mkdir(dname)

    #neural_spike()
    #lagged_phase_space()
    singularity_plot()
