"""
Generate nice plots providing additional information
"""
import sys

import numpy as np

import matplotlib.pylab as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from singularity_finder import compute_tau, compute_singularity_measure, compute_local_phase_field, compute_discrete_gradient, preprocess_data


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
    camp, pacemaker = np.load(sys.argv[1])
    camp = np.rollaxis(camp, 0, 3)
    camp = preprocess_data(camp)

    fig, axarr = plt.subplots(1, 4)
    fig.tight_layout()
    plt.suptitle('pipeline overview')

    def show(data, title, ax):
        ax.set_title(title)
        im = ax.imshow(
            data, interpolation='nearest', cmap=cm.gray)
        holder = make_axes_locatable(ax)
        cax = holder.append_axes('right', size='20%', pad=0.05)
        plt.colorbar(im, cax=cax, format='%.2f')


    rolled_camp = np.rollaxis(camp, 2, 0)
    show(rolled_camp[pos], 'cell overview', axarr[0])

    lphase = compute_local_phase_field(camp)
    show(lphase[pos], 'local phase', axarr[1])

    grad = compute_discrete_gradient(lphase[pos])
    show(grad, 'gradient', axarr[2])

    singularity = compute_singularity_measure(grad)
    show(singularity, 'singularity measure', axarr[3])

    plt.savefig('images/singularity.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    #neural_spike()
    #lagged_phase_space()
    singularity_plot()
