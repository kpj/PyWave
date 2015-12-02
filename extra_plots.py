"""
Generate nice plots providing additional information
"""
import sys

import numpy as np

import matplotlib.pylab as plt
import matplotlib.cm as cm

from singularity_finder import compute_tau, differentiate, compute_singularity_measure


def neural_spike():
    """ Plot various neural spikes
    """
    def do_plot(cell_evo):
        """ Plot [cAMP] for single cell over time
        """
        plt.plot(range(len(cell_evo)), cell_evo)

        plt.title('cAMP concentration in single cell')
        plt.xlabel('t')
        plt.ylabel('cAMP concentration')

        #plt.savefig('images/neural_spike.png', bbox_inches='tight', dpi=300)
        plt.show()

    camp, pacemaker = np.load(sys.argv[1])
    camp = np.rollaxis(camp, 0, 3)

    width, height, depth = camp.shape
    for j in range(width):
        for i in range(height):
            do_plot(camp[i, j])

def lagged_phase_space():
    """ Plot phase space lagged by tau
    """
    def do_plot(cell_evo, i, j):
        x = []
        y = []
        for t in range(len(cell_evo) - tau):
            x.append(cell_evo[t + tau])
            y.append(cell_evo[t])

        plt.plot(x, y)

        plt.title(r'Lagged phase space of neural spike at $x_{%d%d}$ ($\tau = %d$)' % (i, j, tau))
        plt.xlabel(r'$x_{ij}(t - \tau)$')
        plt.ylabel(r'$x_{ij}(t)$')

        #plt.savefig('images/lagged_phase_space.png', bbox_inches='tight', dpi=300)
        plt.show()

    camp, pacemaker = np.load(sys.argv[1])
    camp = np.rollaxis(camp, 0, 3)

    tau = compute_tau(camp)

    width, height, depth = camp.shape
    for j in range(width):
        for i in range(height):
            do_plot(camp[i, j], i, j)

def singularity_plot():
    """ Plot overview over singularity measure
    """
    camp, pacemaker = np.load(sys.argv[1])
    camp = np.rollaxis(camp, 0, 3)

    pos = 1090
    fig, axarr = plt.subplots(1, 3)

    rolled_camp = np.rollaxis(camp, 2, 0)
    axarr[0].set_title('cell overview')
    axarr[0].imshow(rolled_camp[pos], interpolation='nearest', cmap=cm.gray)

    grad = differentiate(camp, pos=pos)
    axarr[1].set_title('gradient')
    axarr[1].imshow(grad, interpolation='nearest', cmap=cm.gray)

    singularity = compute_singularity_measure(camp, pos)
    axarr[2].set_title('singularity measure')
    axarr[2].imshow(singularity, interpolation='nearest', cmap=cm.gray)

    plt.savefig('images/singularity.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    #neural_spike()
    #lagged_phase_space()
    singularity_plot()
