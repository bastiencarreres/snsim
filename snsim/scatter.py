"""This package contains the scattering effects."""

import numpy as np
import sncosmo as snc
import scipy.stats as stats
from . import utils as ut


def init_sn_sct_model(sct_mod, *args):
    """Add scattering effect on sncosmo model.

    Parameters
    ----------
    sct_mod : str
        Name of the model to use.
    Returns
    -------
    None

    """
    if sct_mod == "G10":
        eff_dic = {"source": G10(*args), "name": "G10_", "frame": "rest"}
    elif sct_mod[:3] == "C11":
        eff_dic = {"source": C11, "name": "C11_", "frame": "rest"}
    return eff_dic


class G10(snc.PropagationEffect):
    """Guy (2010) SNe Ia non-coherent scattering.

    Implementation is done following arxiv:1209.2482."""

    _param_names = ["L0", "F0", "F1", "dL", "RndS"]
    param_names_latex = [r"\lambda_0", "F_0", "F_1", "d_L", "RndS"]

    def __init__(self, SALTsource):
        """Initialize G10 class."""
        self._parameters = np.array(
            [2157.3, 0.0, 1.08e-4, 800, np.random.randint(1e11)]
        )
        self._colordisp = SALTsource._colordisp
        self._minwave = SALTsource.minwave()
        self._maxwave = SALTsource.maxwave()

        # self._seed = np.random.SeedSequence()

    def compute_sigma_nodes(self):
        """Computes the sigma nodes."""
        L0, F0, F1, dL = self._parameters[:-1]
        lam_nodes = np.arange(self._minwave, self._maxwave, dL)
        if lam_nodes.max() < self._maxwave:
            lam_nodes = np.append(lam_nodes, self._maxwave)
        siglam_values = self._colordisp(lam_nodes)

        siglam_values[lam_nodes < L0] *= 1 + (lam_nodes[lam_nodes < L0] - L0) * F0
        siglam_values[lam_nodes > L0] *= 1 + (lam_nodes[lam_nodes > L0] - L0) * F1

        return lam_nodes, siglam_values

    def propagate(self, wave, flux):
        """Propagate the effect to the flux."""  # Draw the scattering
        lam_nodes, siglam_values = self.compute_sigma_nodes()
        siglam_values *= np.random.default_rng(int(self._parameters[-1])).normal(
            size=len(lam_nodes)
        )
        magscat = ut.sine_interp(wave, lam_nodes, siglam_values)
        return flux * 10 ** (-0.4 * magscat)


class C11(snc.PropagationEffect):
    """C11 scattering effect for sncosmo.
    Use COV matrix between the vUBVRI bands from N. Chottard thesis.
        Implementation is done following arxiv:1209.2482."""

    _param_names = ["CvU", "Sf", "RndS"]
    param_names_latex = ["\rho_\mathrm{vU}", "S_f", "RndS"]
    _minwave = 2000
    _maxwave = 11000

    def __init__(self):
        """Initialise C11 class."""
        self._parameters = np.array([0.0, 1.3, np.random.randint(1e11)])

        # vUBVRI lambda eff
        self._lam_nodes = np.array([2500.0, 3560.0, 4390.0, 5490.0, 6545.0, 8045.0])

        # vUBVRI correlation matrix extract from SNANA, came from N.Chotard thesis
        self._corr_matrix = np.array(
            [
                [+1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
                [0.000000, +1.000000, -0.118516, -0.768635, -0.908202, -0.219447],
                [0.000000, -0.118516, +1.000000, +0.570333, -0.238470, -0.888611],
                [0.000000, -0.768635, +0.570333, +1.000000, +0.530320, -0.399538],
                [0.000000, -0.908202, -0.238470, +0.530320, +1.000000, +0.490134],
                [0.000000, -0.219447, -0.888611, -0.399538, +0.490134, +1.000000],
            ]
        )

        # vUBVRI sigma
        self._variance = np.array(
            [0.5900, 0.06001, 0.040034, 0.050014, 0.040017, 0.080007]
        )

        # self._seed = np.random.SeedSequence()

    def build_cov(self):
        CvU, Sf = self._parameters[:-1]

        cov_matrix = self._corr_matrix.copy()

        # Set up the vU correlation
        cov_matrix[0, 1:] = CvU * self._corr_matrix[1, 1:]
        cov_matrix[1:, 0] = CvU * self._corr_matrix[1:, 1]

        # Convert corr to cov
        cov_matrix *= np.outer(self._variance, self._variance)

        # Rescale covariance as in arXiv:1209.2482
        cov_matrix *= Sf
        return cov_matrix

    def propagate(self, wave, flux):
        """Propagate the effect to the flux."""

        cov_matrix = self.build_cov()

        # Draw the scattering
        siglam_values = np.random.default_rng(
            int(self._parameters[-1])
        ).multivariate_normal(np.zeros(len(self._lam_nodes)), cov_matrix)

        inf_mask = wave <= self._lam_nodes[0]
        sup_mask = wave >= self._lam_nodes[-1]

        magscat = np.zeros(len(wave))
        magscat[inf_mask] = siglam_values[0]
        magscat[sup_mask] = siglam_values[-1]
        magscat[~inf_mask & ~sup_mask] = ut.sine_interp(
            wave[~inf_mask & ~sup_mask], self._lam_nodes, siglam_values
        )

        return flux * 10 ** (-0.4 * magscat)


##########################################
# GENERATE terms for BS20 scattering model#
##########################################
def gen_BS20_scatter(n_sn, par_names=['beta_sn', 'Rv', 'E_dust', 'c_int'], seed=None):
    """Generate n coherent mag scattering term.

    Parameters
    ----------
    n : int
        Number of mag scattering terms to generate.
    seed : int, optional
        Random seed.

    Returns
    -------
    numpy.ndarray(float)
        numpy array containing scattering terms generated.

    """
    par_names = np.atleast_1d(par_names)
    rand_gen = np.random.default_rng(seed)

    lower, upper = 0.5, 1000
    mu, sigma = 2, 1.4
    Rvdist = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    Rvdist.random_state = rand_gen

    randdist = {'Rv': lambda n: Rvdist.rvs(n),
                'beta_sn': lambda n: rand_gen.normal(
                                    loc=1.98, scale=0.35, # value of mean and sigma are fitted in Brout and Scolnic 2020
                                    size=n
                                    ), 
                'E_dust': lambda n: rand_gen.exponential(
                                    scale=0.1, # value fitted in Brout and Scolnic 2020 shown in table 1
                                    size=n 
                                    ),
                'c_int': lambda n: rand_gen.normal(
                                    loc=-0.084, scale=0.042, # value of mean and sigma are fitted in Brout and Scolnic 2020
                                    size=n
                                    ),
                }
    return [randdist[p](n_sn) for p in par_names]
