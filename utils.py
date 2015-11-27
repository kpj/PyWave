"""
General utilities module
"""

import numpy as np
import numpy.random as npr

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


class Configuration(dict):
    """ Dict with dot-notation access functionality
    """
    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LatticeState(object):
    """ Treat 1D list as 2D lattice and handle coupled system
        This helps with simply passing this object to scipy's odeint
    """
    def __init__(self, width, height, pacemakers=[]):
        """ Initialize lattice
        """
        self.width = width
        self.height = height
        self.pacemakers = pacemakers

        self.discrete_laplacian = np.ones((3, 3)) * 1/2
        self.discrete_laplacian[1, 1] = -4

    def _laplacian(self, i, j, data):
        """ Compute discretized laplacian on Moore neighborhood
        """
        return sum([
            self.discrete_laplacian[k, l] \
            * (data[i+k-1, j+l-1]
                if  i+k-1 < len(data)
                    and j+l-1 < len(data)
                    and i+k-1 >= 0
                    and j+l-1 >= 0
                else 0
            )
                for l in range(3)
                for k in range(3)
        ])

    def _state_matrix(self, i, j, data, threshold=0.8):
        """ Compute state matrix value, with
            refractory cell -> 0
            firing cell -> 1
        """
        if (i, j) in self.pacemakers and npr.random() < config.p:
            return 1
        else:
            return data[i, j] > threshold

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
                camp[i, j] = -config.gamma * camp[i, j] \
                    + config.r * self._state_matrix(i, j, camp) \
                    + config.D * self._laplacian(i, j, camp)
                exci[i, j] = config.eta + config.beta * camp[i, j]

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

def setup_configuration(
        gamma=8, r=300, D=2.3e-7,
        eta=0, beta=0.2,
        p=0.002):
    """ Set model paremeters
    """
    return Configuration({
        'gamma': gamma,
        'r': r,
        'D': D,
        'eta': eta,
        'beta': beta,
        'p': p
    })


# setup model config
global config
config = setup_configuration()
