"""
Find phase singularities, i.e. spiral tips
"""

import sys, os

import numpy as np

from scipy import ndimage
from skimage.draw import circle_perimeter
from skimage.restoration import unwrap_phase

import matplotlib.pylab as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import timed_run, save_data


def compute_tau(camp):
    """ Compute tau as averaged first zero crossing of autocorrelation
    """
    def get_tau(cell_evo):
        """ Get tau for single run
        """
        acorr = np.correlate(cell_evo, cell_evo, mode='full')[len(cell_evo)-1:]

        sign_changes = (np.diff(np.sign(acorr)) != 0)
        if not any(sign_changes):
            raise RuntimeError('No sign change detected')

        tau = sign_changes.nonzero()[0][0]
        return tau

    width, height, depth = camp.shape
    tau_list = []
    for j in range(width):
        for i in range(height):
            try:
                tau_list.append(get_tau(camp[i, j]))
            except RuntimeError:
                # TODO: what happens in this case?
                pass

    assert len(tau_list) > 0, 'No sign changes found'
    return int(np.mean(tau_list))

def compute_phase_variable(cell_evo, tau):
    """ Generate local phase space
    """
    m = np.mean(cell_evo)
    theta = []
    for t in range(len(cell_evo) - tau):
        cur = np.arctan2(
            cell_evo[t + tau] - m,
            cell_evo[t] - m
        )
        theta.append(cur)

    return theta

@timed_run('Computing gradients')
def compute_discrete_gradient(fields):
    """ Compute discretized gradient on given field
    """
    gradients = []
    for field in fields:
        # finite difference operator
        fidi_op_x = np.diff(field, axis=0)
        fidi_op_x = np.vstack((fidi_op_x, fidi_op_x[-1]))
        fidi_op_y = np.diff(field, axis=1)
        fidi_op_y = np.hstack((fidi_op_y, fidi_op_y[:,-1][np.newaxis].T))

        # compute convolution
        nabla_x = np.array([
            [-1/2, 0, 1/2],
            [-1, 0, 1],
            [-1/2, 0, 1/2]
        ])
        nabla_y = np.array([
            [1/2, 1, 1/2],
            [0, 0, 0],
            [-1/2, -1, -1/2]
        ])

        conv_x = ndimage.convolve(
            fidi_op_y, nabla_x,
            mode='constant', cval=0.0
        )
        conv_y = ndimage.convolve(
            fidi_op_x, nabla_y,
            mode='constant', cval=0.0
        )

        # compute discretized gradient
        grad = conv_x + conv_y
        gradients.append(grad)

    return np.array(gradients)

@timed_run('Computing local phase field')
def compute_local_phase_field(camp):
    """ Compute local phase of each cell
    """
    width, height, depth = camp.shape

    # compute thetas
    tau = compute_tau(camp)

    theta = np.empty((width, height)).tolist()
    for j in range(width):
        for i in range(height):
            cur = compute_phase_variable(camp[i, j], tau)
            theta[i][j] = np.array(cur)
    theta = np.rollaxis(np.array(theta), 2, 0)

    # fix phase jumps greater than PI
    for i in range(len(theta)):
        theta[i] = unwrap_phase(theta[i])

    return theta

@timed_run('Computing singularity measure')
def compute_singularity_measure(gradients):
    """ Compute singularity measure of data
    """
    # precompute closed curve coordinates
    circle_rad = 5

    grad_dim = gradients[0].shape
    closed_curves = np.empty(grad_dim, dtype=(list, list))
    for j in range(grad_dim[1]):
        for i in range(grad_dim[0]):
            rr, cc = circle_perimeter(
                i, j, circle_rad,
                method='andres', shape=grad_dim
            )
            closed_curves[i, j] = (rr, cc)

    # find singularities
    singularities = []
    for grad in gradients:
        width, height = grad.shape

        singularity = np.empty((width, height))
        for j in range(height):
            for i in range(width):
                res = np.sum(grad[closed_curves[i, j]])
                singularity[i, j] = res
        singularity = np.array(singularity)

        singularities.append(singularity)

    return np.array(singularities)

@timed_run('Preprocessing data')
def preprocess_data(data):
    """ Preprocess data to make singularity detection easier
    """
    # eliminate background
    data = np.diff(data)

    # average elements
    data = np.array([0.5*(data[...,i]+data[...,i+1]) for i in range(0, data.shape[2]-1, 2)])
    data = np.rollaxis(data, 0, 3)

    # apply gaussian filter
    def gf(field):
        sig = 5
        nrows, ncols = field.shape
        return ndimage.gaussian_filter(
            field,
            sigma=(sig * nrows / 100.0, sig * ncols / 100.0),
            order=0
        )
    data = np.array([gf(data[...,i]) for i in range(data.shape[2])])
    data = np.rollaxis(data, 0, 3)

    return data

def singularity_plot(
    pure_fname,
    rolled_camp, lphase, grads,
    singularities, avg_singularity, thres_singularity):
    """ Plot overview over singularity measure
    """
    # handle img directory
    img_dir = 'images'
    if not os.path.isdir(os.path.join(img_dir, pure_fname)):
        os.makedirs(os.path.join(img_dir, pure_fname))

    # plot data
    pos_num = 4
    pos_range = range(0, lphase.shape[0], int(lphase.shape[0]/(pos_num)))[1:]

    fig, axarr = plt.subplots(len(pos_range), 4, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle('pipeline overview')

    def show(data, title, ax):
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        im = ax.imshow(
            data, interpolation='nearest', cmap=cm.gray)

        holder = make_axes_locatable(ax)
        cax = holder.append_axes('right', size='20%', pad=0.05)
        plt.colorbar(im, cax=cax, format='%.2f')

    for axrow, pos in enumerate(pos_range):
        show(rolled_camp[pos], 'cell overview', axarr[axrow][0])
        show(lphase[pos], 'local phase', axarr[axrow][1])
        show(grads[pos], 'gradient', axarr[axrow][2])
        show(singularities[pos], 'singularity measure', axarr[axrow][3])

    plt.savefig(os.path.join(img_dir, pure_fname, 'singularity.png'), bbox_inches='tight', dpi=300)
    #plt.show()

    # averaged results
    fig, axarr = plt.subplots(1, 2, figsize=(10, 10))

    show(avg_singularity, 'averaged singularity measure', axarr[0])
    show(thres_singularity, 'thresholded singularity measure', axarr[1])

    plt.savefig(os.path.join(img_dir, pure_fname, 'averaged_singularity.png'), bbox_inches='tight', dpi=300)
    #plt.show()

def compute_spiral_tip_density(fname, plot=True):
    """ Compute spiral tip density of given example
    """
    cache_dir = 'cache'
    pure_fname = os.path.splitext(os.path.basename(fname))[0]

    if not os.path.isdir(os.path.join(cache_dir, pure_fname)):
        # preprocess input
        camp, pacemaker, used_config = np.load(fname)
        camp = np.rollaxis(camp, 0, 3)
        camp = preprocess_data(camp)

        # compute data
        rolled_camp = np.rollaxis(camp, 2, 0)
        lphase = compute_local_phase_field(camp) # decreases time dim due to tau
        grads = compute_discrete_gradient(lphase)
        singularities = compute_singularity_measure(grads)

        # cache data
        os.path.join(cache_dir, pure_fname, '')
        save_data(os.path.join(cache_dir, pure_fname, 'rolled_camp'), rolled_camp)
        save_data(os.path.join(cache_dir, pure_fname, 'lphase'), lphase)
        save_data(os.path.join(cache_dir, pure_fname, 'grads'), grads)
        save_data(os.path.join(cache_dir, pure_fname, 'singularities'), singularities)
    else:
        print(' > Using cached data')
        rolled_camp = np.load(os.path.join(cache_dir, pure_fname, 'rolled_camp.npy'))
        lphase = np.load(os.path.join(cache_dir, pure_fname, 'lphase.npy'))
        grads = np.load(os.path.join(cache_dir, pure_fname, 'grads.npy'))
        singularities = np.load(os.path.join(cache_dir, pure_fname, 'singularities.npy'))

    # compute singularity measures
    avg_singularity = np.mean(singularities, axis=0)

    thres_singularity = avg_singularity.copy()
    thres_singularity[thres_singularity > np.pi] = 2 * np.pi
    thres_singularity[thres_singularity < -np.pi] = -2 * np.pi
    thres_singularity[(thres_singularity > -np.pi) & (thres_singularity < np.pi)] = 0

    # plot data if needed
    if plot:
        singularity_plot(
            pure_fname,
            rolled_camp, lphase, grads,
            singularities, avg_singularity, thres_singularity
        )

    # compute tip density
    tip_num = np.count_nonzero(thres_singularity)
    tip_density = float(tip_num) / thres_singularity.size

    return tip_density


def main():
    """ Detect phase singularity density and create overview plots
    """
    if len(sys.argv) != 2:
        print('Usage: %s <data file/dir>' % sys.argv[0])
        sys.exit(1)

    arg = sys.argv[1]
    if os.path.isdir(arg):
        for fn in os.listdir(arg):
            fname = os.path.join(arg, fn)

            density = compute_spiral_tip_density(fname)
            print(fn, density)
    else:
        density = compute_spiral_tip_density(arg)
        print(density)

if __name__ == '__main__':
    main()
