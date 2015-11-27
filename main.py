"""
Reproduction of figure from Sawai et al. (2005)
"""

import numpy as np
from scipy.integrate import odeint

from utils import LatticeState, animate_evolution


def generate_system(i, j):
    """ Setup lattice of size `i`x`j`
    """
    system = LatticeState(i, j)
    print(system)

    return system

def integrate_system(system):
    """ Integrate ODE-CA hybrid.
        Levine et al. (1996)
    """
    def func(state, t):
        return system.get_ode(state, t)

    init = [0.5] * system.get_size() * 2
    t_range = np.arange(0, 5, 0.1)

    res = odeint(func, init, t_range)
    camp_res, exci_res = system.parse_result(res)

    return camp_res

def plot_system(system):
    """ Visualize system state over time
    """
    # roll axes to make time-access easier
    rolled_system = np.rollaxis(system, 2)

    # create animation
    animate_evolution(rolled_system)

def main():
    """ Main interface
    """
    system = generate_system(2, 2)
    cres = integrate_system(system)
    plot_system(cres)


if __name__ == '__main__':
    main()
