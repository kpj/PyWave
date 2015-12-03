"""
Find phase singularities, i.e. spiral tips
"""

import sys

import numpy as np
from scipy import ndimage
from skimage.draw import circle_perimeter


## for fixed ij
# input: x_ij(t)
# create phase space: x_ij(t+tau) vs x_ij(t)
#   where tau = first zero crossing of autocorrelation of x_ij(t)
# extract: T_ij(t) = atan( x_ij(t+tau)-x_ij_mean, x_ij(t)-x_ij_mean )

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

def compute_discrete_gradient(field):
    """ Compute discretized gradient on given field
    """
    # fix phase jumps greater than PI
    for i in range(field.shape[0]):
        field[i] = np.unwrap(field[i])

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
    return grad

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

    return theta

def integrate(cx, cy, radius, grad):
    """ Integrate gradient along circle perimeter of given shape
    """
    rr, cc = circle_perimeter(
        cx, cy, radius,
        method='andres', shape=grad.shape
    )
    res = sum(grad[rr, cc])
    return res

def compute_singularity_measure(grad):
    """ Compute singularity measure of data
    """
    width, height = grad.shape

    circle_rad = 3
    singularity = np.empty((width, height))
    for j in range(height):
        for i in range(width):
            res = integrate(i, j, circle_rad, grad)
            singularity[i, j] = res
    singularity = np.array(singularity)

    return singularity

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


def main(data):
    """ Detect phase singularities
    """
    pos = 0

    print(data.shape)
    data = preprocess_data(data)

    lphase = compute_local_phase_field(data)
    grad = compute_discrete_gradient(lphase[pos])

    singularity = compute_singularity_measure(grad)
    print(singularity)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <data file>' % sys.argv[0])
        sys.exit(1)

    data, pacemaker = np.load(sys.argv[1])
    data = np.rollaxis(data, 0, 3)
    main(data)
