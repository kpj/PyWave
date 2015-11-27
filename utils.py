"""
General utilities module
"""

import numpy as np

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


class LatticeState(object):
    """ Treat 1D list as 2D lattice and handle coupled system
        This helps with simply passing this object to scipy's odeint
    """
    def __init__(self, width, height):
        """ Initialize lattice
        """
        self.width = width
        self.height = height

    def get_size(self):
        """ Return number of cells in underlying system
        """
        return self.width * self.height

    def get_ode(self, state, t):
        """ Return corresponding ODE
            Structure:
            [
                camp00, camp01, .. ,camp0m, camp10, .., campnm
                ...
                exci00, exci01, .. ,exci0m, exci10, .., excinm
            ]
        """
        camp = np.zeros((self.width, self.height))
        exci = np.zeros((self.width, self.height))

        for j in range(self.width):
            for i in range(self.height):
                camp[i, j] = i*j
                exci[i, j] = i+j

        flat_camp = np.reshape(camp, self.get_size())
        flat_exci = np.reshape(exci, self.get_size())

        return np.append(flat_camp, flat_exci)

    def parse_result(self, orig_res):
        """ Parse integration result
        """
        t_range = len(orig_res)
        res = orig_res.T

        flat_camp = res[:self.get_size()].reshape(self.get_size() * t_range)
        flat_exci = res[self.get_size():].reshape(self.get_size() * t_range)

        camp = np.reshape(flat_camp, (self.width, self.height, t_range))
        exci = np.reshape(flat_exci, (self.width, self.height, t_range))

        return camp, exci

    def __repr__(self):
        """ Nice visual representation of lattice
        """
        return '%dx%d' % (self.width, self.height)

def animate_evolution(system, fname='lattice.gif'):
    """ Animation evolution of lattice over time
    """
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    im = plt.imshow(
        system[0],
        cmap=cm.gray, interpolation='nearest',
        vmin=np.amin(system), vmax=np.amax(system)
    )
    plt.colorbar(im)

    def update(t):
        plt.suptitle(r'$t = %d$' % t)
        im.set_data(system[t])
        return im,

    ani = animation.FuncAnimation(
        plt.gcf(), update,
        frames=len(system)
    )

    ani.save(fname, writer='imagemagick', fps=10)#, dpi=200)
    plt.close()
