"""
Generate nice plots providing additional information
"""
import sys

import numpy as np
import matplotlib.pylab as plt


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


if __name__ == '__main__':
    neural_spike()
