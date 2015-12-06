"""
Conduct experiment which aims to reproduce figure 4a) from Sawai et al. (2005)
"""

import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pylab as plt

from main import run_system
from configuration import get_config
from lattice_initializer import Generator
from singularity_finder import compute_spiral_tip_density
from utils import save_data


def handle_run_system(beta):
    """ Handle computations for single beta
    """
    config = get_config(beta=beta)
    run_system(Generator)

def handle_measure_tips(fname):
    """ Compute tip density for given data
    """
    camp, pacemaker, used_config = np.load(fname)
    density = compute_spiral_tip_density(fname, plot=False)

    return (used_config['beta'], density)

def generate_data(resolution=10):
    """ Generate simulation data and save it in data-files
    """
    log_vals = np.logspace(-4, 0, num=resolution, endpoint=False)
    with Pool(resolution) as p:
        p.map(handle_run_system, log_vals)

def measure_tips(out_fname='results/experiment_run'):
    """ Compute spiral-tip density for all available data files
    """
    data_dir = 'data'
    files = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]

    with Pool(len(files)) as p:
        data = p.map(handle_measure_tips, files)

    save_data(out_fname, data)

def plot_result(fname='results/experiment_run.npy'):
    """ Plot beta/density overview
    """
    data = np.load(fname)

    plt.semilogx(*zip(*data), marker='o', ls='')

    plt.title(r'Influence of $\beta$ on spiral-tip density')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'spiral-tip density')

    img_dir = 'images'
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    plt.savefig(
        os.path.join(img_dir, 'beta_overview.png'),
        bbox_inches='tight', dpi=300)

def main():
    """ Simulate model for varying beta
    """
    generate_data()
    measure_tips()
    plot_result()


if __name__ == '__main__':
    main()
