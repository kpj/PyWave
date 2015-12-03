"""
Generate nice plots providing additional information
"""
import sys

import numpy as np

import matplotlib.pylab as plt
import matplotlib.cm as cm

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

    pos = 0
    fig, axarr = plt.subplots(1, 4)
    plt.suptitle('pipeline overview')

    rolled_camp = np.rollaxis(camp, 2, 0)
    axarr[0].set_title('cell overview')
    axarr[0].imshow(rolled_camp[pos], interpolation='nearest', cmap=cm.gray)

    lphase = compute_local_phase_field(camp)
    axarr[1].set_title('local phase')
    phase_im = axarr[1].imshow(lphase[pos], interpolation='nearest', cmap=cm.gray)

    grad = compute_discrete_gradient(lphase[pos])
    axarr[2].set_title('gradient')
    axarr[2].imshow(grad, interpolation='nearest', cmap=cm.gray)

    singularity = compute_singularity_measure(grad)
    axarr[3].set_title('singularity measure')
    axarr[3].imshow(
        singularity, interpolation='nearest', cmap=cm.gray)

    plt.savefig('images/singularity.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    #neural_spike()
    #lagged_phase_space()
    singularity_plot()
