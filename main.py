"""
Reproduction of figure from Sawai et al. (2005)
"""

import numpy as np

from progressbar import ProgressBar

from configuration import get_config
from lattice_initializer import Generator
from utils import animate_evolution


def integrate_system(system):
    """ Integrate ODE-CA hybrid with simple Euler method
        Levine et al. (1996)
    """
    t = 0
    state = Generator.get_initial_state(system)

    pbar = ProgressBar(maxval=config.t_max)
    data = []
    pbar.start()
    while t < config.t_max:
        data.append(state)
        state += config.dt * system.get_ode(state, t)
        t += config.dt
        pbar.update(int(t))
    pbar.finish()

    camp_res, exci_res = system.parse_result(np.array(data))
    return camp_res

def plot_system(system, pacemakers):
    """ Visualize system state over time
    """
    # roll axes to make time-access easier
    rolled_system = np.rollaxis(system, 2)
    np.save('save', np.array([rolled_system, pacemakers]))

    # create animation
    animate_evolution(rolled_system, pacemakers)

def main():
    """ Main interface
    """
    global config
    config = get_config()

    system = Generator(config.grid_size).generate()
    print(system)

    cres = integrate_system(system)
    plot_system(cres, system.pacemakers)


if __name__ == '__main__':
    main()
