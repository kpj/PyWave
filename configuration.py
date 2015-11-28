"""
Store free parameters of model
"""


class Configuration(dict):
    """ Dict with dot-notation access functionality
    """
    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def setup_configuration(
        gamma=8, r=300, D=2.3e-7,
        eta=0, beta=0.2,
        p=0.002,
        c_min=4, c_max=100,
        t_arp=2, t_rrp=7, t_f=1,
        e_max=0.93,
        dt=0.01, t_max=1000, grid_size=100):
    """ Set model paremeters
    """
    return Configuration({
        'gamma': gamma, 'r': r, 'D': D,
        'eta': eta, 'beta': beta,
        'p': p,
        'c_min': c_min, 'c_max': c_max,
        't_arp': t_arp, 't_rrp': t_rrp, 't_f': t_f,
        'e_max': e_max,
        'dt': dt, 't_max': t_max, 'grid_size': grid_size
    })


# setup model config
global config
def get_config():
    return config

config = setup_configuration()
