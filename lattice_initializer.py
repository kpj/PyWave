"""
Setup different initial conditions for the lattice
"""

import numpy as np
import numpy.random as npr

from configuration import get_config
from model import LatticeState


class BaseGenerator(object):
    def __init__(self, i, j=None):
        """ Setup lattice of size `i`x`j`
        """
        if j is None:
            j = i

        self.width = i
        self.height = j

    def generate(self):
        """ Return lattice object
        """
        raise NotImplementedError('No lattice is being generated')

    @classmethod
    def get_initial_state(cls, system):
        """ Return lattice object
        """
        raise NotImplementedError('No initial state is being generated')

class Default(BaseGenerator):
    def generate(self):
        pacemakers = []
        for _ in range(self.width):
            pacemakers.append((
                npr.randint(0, self.width),
                npr.randint(0, self.width)
            ))

        return LatticeState(self.width, self.height, pacemakers=pacemakers)

    @classmethod
    def get_initial_state(cls, system):
        return [0] * system.get_size() * 2

class SingleSpiral(BaseGenerator):
    def generate(self):
        return LatticeState(self.width, self.height)

    @classmethod
    def get_initial_state(cls, system):
        w, h = system.width, system.height

        camp = np.zeros((w, h))
        exci = np.zeros((w, h))

        for j in range(int(w/5)):
            camp[int(h/2), j] = config.c_max

        return system._camp_exci2state_vec(camp, exci)


config = get_config()
Generator = Default
