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

    # make animation smaller and faster
    states = states[::10]

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
        print('Usage: %s <data file/dir>' % sys.argv[0])
        sys.exit(1)

    def get_savedir(arg):
        """ Figure out where to save images for this data file and
            make sure the directory actually exists
        """
        img_dir = 'images'
        pure_fname = os.path.splitext(os.path.basename(arg))[0]
        save_dir = os.path.join(img_dir, pure_fname)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        return save_dir

    arg = sys.argv[1]
    if os.path.isdir(arg):
        for fn in os.listdir(arg):
            fname = os.path.join(arg, fn)
            camp, pacemaker, used_config = np.load(fname)

            out_name = os.path.join(get_savedir(fname), 'lattice_evolution.gif')
            animate_evolution(camp, pacemaker, fname=out_name)
    else:
        camp, pacemaker, used_config = np.load(arg)

        out_name = os.path.join(get_savedir(arg), 'lattice_evolution.gif')
        animate_evolution(camp, pacemaker, fname=out_name)
