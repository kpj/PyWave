"""
Reproduction of figure from Sawai et al. (2005)
"""

import numpy as np
import numpy.random as npr

from progressbar import ProgressBar

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
    """ Integrate ODE-CA hybrid with simple Euler method
        Levine et al. (1996)
    """
    t = 0
    state = [0] * system.get_size() * 2

    pbar = ProgressBar(maxval=config.t_max)
    data = []
    pbar.start()
    while t < config.t_max:
        data.append(state)
        state = state + config.dt * system.get_ode(state, t)
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

    system = generate_system(config.grid_size)
    cres = integrate_system(system)
    plot_system(cres, system.pacemakers)


if __name__ == '__main__':
    main()
