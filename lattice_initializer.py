"""
Setup different initial conditions for the lattice
"""

import numpy.random as npr

from utils import LatticeState


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

class Default(BaseGenerator):
    def generate(self):
        pacemakers = []
        for _ in range(self.width):
            pacemakers.append((
                npr.randint(0, self.width),
                npr.randint(0, self.width)
            ))

        return LatticeState(self.width, self.height, pacemakers=pacemakers)


Generator = Default
