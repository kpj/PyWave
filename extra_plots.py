"""
Generate nice plots providing additional information
"""
import sys, os

import numpy as np

import matplotlib.pylab as plt


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


if __name__ == '__main__':
    dname = 'images'
    if not os.path.isdir(dname):
        os.mkdir(dname)

    #neural_spike()
    lagged_phase_space()
