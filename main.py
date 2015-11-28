"""
Reproduction of figure from Sawai et al. (2005)
"""

import numpy as np
import numpy.random as npr
from scipy.integrate import odeint

from utils import get_config, LatticeState, animate_evolution


def generate_system(i, j=None):
    """ Setup lattice of size `i`x`j`
    """
    if j is None:
        j = i

    pacemakers = []
    for _ in range(i):
        pacemakers.append((
            npr.randint(0, i),
            npr.randint(0, j)
        ))

    system = LatticeState(i, j, pacemakers=pacemakers)
    print(system)

    return system

def integrate_system(system):
    """ Integrate ODE-CA hybrid.
        Levine et al. (1996)
    """
    def func(state, t):
        return system.get_ode(state, t)

    init = [0] * system.get_size() * 2
    t_range = np.arange(0, config.t_max, config.dt)

    res = odeint(func, init, t_range)
    camp_res, exci_res = system.parse_result(res)

    return camp_res

def plot_system(system, pacemakers):
    """ Visualize system state over time
    """
    # roll axes to make time-access easier
    rolled_system = np.rollaxis(system, 2)

    # create animation
    animate_evolution(rolled_system, pacemakers)

def main():
    """ Main interface
    """
    global config
    config = get_config()

    system = generate_system(4)
    cres = integrate_system(system)
    plot_system(cres, system.pacemakers)


if __name__ == '__main__':
    main()
