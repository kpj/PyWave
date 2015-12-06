"""
General utilities module
"""

import os, time

import numpy as np

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

from configuration import get_config


config = get_config()

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
    if len(pacemakers) > 0:
        plt.scatter(
            *zip(*[reversed(p) for p in pacemakers]),
            marker='+', color='red'
        )

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

def save_data(fname, data):
    """ Try to save `data` in `directory`. Create `directory` if it does not exist
    """
    dname = os.path.dirname(os.path.abspath(fname))
    if not os.path.isdir(dname):
        os.makedirs(dname)
    np.save(fname, data)

def gen_run_identifier():
    """ Extract config parameters which are likely to be distinctive.
        Note: this requires a properly setup configuration
    """
    return ('data_%.4f_%d_%d_%.2f_%.3f__%s' % ( \
        config.beta,
        config.grid_size, config.t_max,
        config.D, config.p,
        time.strftime('%Y%m%d%H%M%S'))).replace('.', 'p')

def timed_run(title):
    """ Decorator which times the decorated function
    """
    def tmp(func):
        def wrapper(*args, **kwargs):
            print(' > %s' % title, end=' ', flush=True)
            start_time = time.time()
            res = func(*args, **kwargs)
            run_dur = time.time() - start_time
            print('(%.2fs)' % run_dur)
            return res
        return wrapper
    return tmp


# plot data from previous run
if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print('Usage: %s <data file>' % sys.argv[0])
        sys.exit(1)

    data = np.load(sys.argv[1])
    animate_evolution(data[0], data[1])
