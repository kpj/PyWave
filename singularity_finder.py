"""
Find phase singularities, i.e. spiral tips
"""

import sys

import numpy as np
from scipy import ndimage


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
    # convolution matrices
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

    # finite difference operator
    # TODO: fix phase jumps greater than PI
    fidi_op_x = np.diff(field, axis=0)
    fidi_op_x = np.vstack((fidi_op_x, field[-1]))
    fidi_op_y = np.diff(field, axis=1)
    fidi_op_y = np.hstack((fidi_op_y, field[:,-1][np.newaxis].T))

    # compute convolution
    conv_x = ndimage.convolve(
        fidi_op_x, nabla_x,
        mode='constant', cval=0.0
    )
    conv_y = ndimage.convolve(
        fidi_op_y, nabla_y,
        mode='constant', cval=0.0
    )

    # compute discretized gradient
    grad = conv_x + conv_y
    return grad

def differentiate(camp):
    """ Compute discretized nabla of local phase
    """
    width, height, depth = camp.shape

    # compute thetas
    tau = compute_tau(camp)
    print('Tau:', tau)

    theta = np.empty((width, height)).tolist()
    for j in range(width):
        for i in range(height):
            cur = compute_phase_variable(camp[i, j], tau)
            theta[i][j] = np.array(cur)
    theta = np.rollaxis(np.array(theta), 2, 0)

    # infer gradient
    t_theta = theta[1]
    grad = compute_discrete_gradient(t_theta)

    return grad

def main(data):
    """ Detect phase singularities
    """
    print(data.shape)
    grad = differentiate(data)
    print(grad.shape)
    print(grad)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <data file>' % sys.argv[0])
        sys.exit(1)

    data, pacemaker = np.load(sys.argv[1])
    data = np.rollaxis(data, 0, 3)
    main(data)
