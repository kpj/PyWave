"""
Reproduction of figure from Sawai et al. (2005)
"""

import numpy as np

from progressbar import ProgressBar

from configuration import get_config
from lattice_initializer import Generator
from utils import animate_evolution, save_data, gen_run_identifier


config = get_config()

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
        state = state + config.dt * system.get_ode(state, t)
        t += config.dt
        pbar.update(int(t))
    pbar.finish()

    camp_res, exci_res = system.parse_result(np.array(data))
    return np.rollaxis(camp_res, 2) # roll axes to make time-access easier

def run_system(Generator):
    """ Apply `Generator` and integrate and cache system
    """
    system = Generator(config.grid_size).generate()
    print(system)

    cres = integrate_system(system)

    fname = gen_run_identifier()
    save_data(
        'data/%s' % fname,
        np.array([cres, system.pacemakers, dict(config)]))

    return system, cres

def main():
    """ Main interface
    """
    system, cres = run_system(Generator)
    animate_evolution(cres, system.pacemakers)


if __name__ == '__main__':
    main()
