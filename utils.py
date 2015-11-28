"""
General utilities module
"""

import numpy as np
import numpy.random as npr

from scipy import ndimage

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

from configuration import get_config


config = get_config()

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

        self.state_matrix = np.zeros((width, height))
        self.tau_matrix = np.ones((width, height)) * (-config.t_arp) # in ARP

    def _update_state_matrix(self, camp, exci):
        """ Compute state matrix value, with
            quiescent/refractory cell -> 0
            firing cell -> 1
        """
        # this function gets executed once per timestep
        for j in range(self.width):
            for i in range(self.height):
                if self.state_matrix[i, j] == 0: # not firing
                    self.handle_off_cell(i, j, camp, exci)
                else: # firing
                    self.handle_on_cell(i, j)

    def handle_on_cell(self, i, j):
        """ Handle cell where state_matrix == 1
        """
        self.tau_matrix[i, j] += config.dt

        if self.tau_matrix[i, j] >= 0: # end of firing reached
            self.state_matrix[i, j] = 0
            self.tau_matrix[i, j] = -config.t_arp

    def handle_off_cell(self, i, j, camp, exci):
        """ Handle cell where state_matrix == 0
        """
        tau = self.tau_matrix[i, j]

        if tau >= 0: # in RRP
            A = ((config.t_rrp + config.t_arp) \
                * (config.c_max - config.c_min)) / config.t_rrp
            t = (config.c_max - A * (tau / (tau + config.t_arp))) \
                * (1 - exci[i, j])

            # increase time up to t_rrp
            if tau < config.t_rrp:
                self.tau_matrix[i, j] += config.dt

            # check threshold
            if camp[i, j] > t:
                self.fire_cell(i, j)

            # handle pacemaker
            if (i, j) in self.pacemakers and npr.random() < config.p:
                self.fire_cell(i, j)
        else: # in ARP
            self.tau_matrix[i, j] += config.dt

    def fire_cell(self, i, j):
        """ Fire cell `i`x`j`
        """
        self.state_matrix[i, j] = 1
        self.tau_matrix[i, j] = -config.t_f

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
        # parse ODE state
        flat_camp = state[:self.get_size()]
        flat_exci = state[self.get_size():]

        camp = np.reshape(flat_camp, (self.width, self.height))
        exci = np.reshape(flat_exci, (self.width, self.height))

        # compute next iteration
        self._update_state_matrix(camp, exci)

        next_camp = np.zeros((self.width, self.height))
        next_exci = np.zeros((self.width, self.height))

        laplacian_conv = ndimage.convolve(
            camp, self.discrete_laplacian,
            mode='constant', cval=0.0
        )

        for j in range(self.width):
            for i in range(self.height):
                next_camp[i, j] = -config.gamma * camp[i, j] \
                    + config.r * self.state_matrix[i, j] \
                    + config.D * laplacian_conv[i, j]

                if exci[i, j] < config.e_max:
                    next_exci[i, j] = config.eta + config.beta * camp[i, j]

        flat_camp = np.reshape(next_camp, self.get_size())
        flat_exci = np.reshape(next_exci, self.get_size())

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

def animate_evolution(states, pacemakers, fname='lattice.gif'):
    """ Animation evolution of lattice over time
    """
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    im = plt.imshow(
        states[0],
        cmap=cm.gray, interpolation='nearest',
        vmin=np.amin(states), vmax=np.amax(states)
    )
    plt.colorbar(im)
    plt.scatter(*zip(*[reversed(p) for p in pacemakers]), marker='+', color='red')

    def update(t):
        plt.suptitle(r'$t = %d$' % t)
        im.set_data(states[t])
        return im,

    ani = animation.FuncAnimation(
        plt.gcf(), update,
        frames=len(states)
    )

    ani.save(fname, writer='imagemagick', fps=10)#, dpi=200)
    plt.close()


# plot data from previous run
if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print('Usage: %s <data file>' % sys.argv[0])
        sys.exit(1)

    data = np.load(sys.argv[1])
    animate_evolution(data[0], data[1])
