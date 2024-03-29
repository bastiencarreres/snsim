"""This package contains the scattering effects."""

import numpy as np
import sncosmo as snc
from . import nb_fun as nbf


def init_sn_sct_model(model, sct_mod):
    """Add scattering effect on sncosmo model.

    Parameters
    ----------
    model : sncosmo.Model
        The model on which add effects.
    sct_mod : str
        Name of the model to use.

    Returns
    -------
    None

    """
    if sct_mod == 'G10':
        model.add_effect(G10(model), 'G10_', 'rest')

    elif sct_mod[:3] == 'C11':
        model.add_effect(C11(model), 'C11_', 'rest')
        if sct_mod == 'C11_1':
            model.set(C11_Cuu=1.)
        elif sct_mod == 'C11_2':
            model.set(C11_Cuu=-1.)


class G10(snc.PropagationEffect):
    """G10 scattering effect for sncosmo.

    Parameters
    ----------
    model : sncosmo.Model
        The sncosmo Model of the SN.

    Attributes
    ----------
    _parameters : list
        List containing all the model parameters.
    _minwave : float
        The minimal wavelength of the effect.
    _maxwave : float
        The maximal wavelength of the effect.
    _colordisp : function
        The color dispersion of SALT model.
    _param_names : list(str)
        Names of the parameters.
    param_names_latex : list(str)
        Latex version of parameters names.

    Notes
    -----
    Use colordisp file of salt and follow SNANA formalism, see arXiv:1209.2482

    """

    _param_names = ['L0', 'F0', 'F1', 'dL', 'RndS']
    param_names_latex = [r'\lambda_0', 'F_0', 'F_1', 'd_L', 'RS']

    def __init__(self, model):
        """Initialize G10 class."""
        self._parameters = np.array([2157.3, 0.0, 1.08e-4, 800,
                                     np.random.randint(low=1000, high=100000)])
        self._minwave = model.source.minwave()
        self._maxwave = model.source.maxwave()
        self._colordisp = model.source._colordisp

    @property
    def scattering_law(self):
        """Construct the scattering law.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            wavelength, scatter law at wavelength.

        """
        L0, F0, F1, dL = self._parameters[:-1]
        lam = self._minwave
        sigma_lam = []
        sigma_val = []

        while lam < self._maxwave:
            sigma_lam.append(lam)
            val = self._colordisp(lam)
            if lam > L0:
                val *= 1 + (lam - L0) * F1
            elif lam < L0:
                val *= 1 + (lam - L0) * F0

            sigma_val.append(val)
            lam += dL
        return np.asarray(sigma_lam), np.asarray(sigma_val)

    @property
    def lam_scatter(self):
        """Generate the scatter.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            wavelength, random scatter at wavelength.

        """
        sigma_lam, sigma_val = self.scattering_law
        RS = self._parameters[-1]
        return sigma_lam, sigma_val * \
            np.random.default_rng(int(RS)).normal(0, 1, size=len(sigma_val))

    def propagate(self, wave, flux):
        """Propagate the effect to the flux.

        Parameters
        ----------
        wave : float
            wavelength.
        flux : float
            flux density at wavelength.

        Returns
        -------
        numpy.ndarray(float)
            Flux density with effect applied.
        """
        lam, scatter = self.lam_scatter
        scattering = np.asarray([nbf.sine_interp(w, lam, scatter) for w in wave])
        return flux * 10**(-0.4 * scattering)


class C11(snc.PropagationEffect):
    """C11 scattering effect for sncosmo.

    Parameters
    ----------
    model : sncosmo.Model
        The sncosmo Model of the SN.

    Attributes
    ----------
    _parameters : list
        List containing all the model parameters.
    _minwave : float
        The minimal wavelength of the effect.
    _maxwave : float
        The maximal wavelength of the effect.
    _sigma_lam : numpy.ndarray(float, size = 6)
        Value of the effective wavelengths of U'UBVRI bands.
    _CORR_matrix : numpy.ndarray(float, sizee = (6,6))
        Correlation matrix of U'UBVRI bands scattering.
    _sigma : numpy.ndarray(float, size = 6)
        Mean scattering in U'UBVRI bands.
    _param_names : list(str)
        Names of the parameters.
    param_names_latex : list(str)
        Latex version of parameters names.

    Notes
    -----
    Use COV matrix from N. Chottard thesis and follow SNANA formalism, see arXiv:1209.2482

    """

    _param_names = ['Cuu', 'Sc', 'RndS']
    param_names_latex = ["\rho_{u'u}", 'Sc', 'RS']

    def __init__(self, model):
        """Initialise C11 class."""
        self._parameters = np.array([0., 1.3, np.random.randint(low=1000, high=100000)])
        self._minwave = model.source.minwave()
        self._maxwave = model.source.maxwave()

        # U'UBVRI lambda eff
        self._sigma_lam = np.array([2500.0, 3560.0, 4390.0, 5490.0, 6545.0, 8045.0])
        # U'UBVRI correlation matrix extract from SNANA, came from N.Chotard thesis
        self._corr_matrix = np.array(
            [[+1.000, 0.000, 0.000, 0.000, 0.000, 0.000],
             [0.000, +1.000000, -0.118516, -0.768635, -0.908202, -0.219447],
             [0.000, -0.118516, +1.000000, +0.570333, -0.238470, -0.888611],
             [0.000, -0.768635, +0.570333, +1.000000, +0.530320, -0.399538],
             [0.000, -0.908202, -0.238470, +0.530320, +1.000000, +0.490134],
             [0.000, -0.219447, -0.888611, -0.399538, +0.490134, +1.000000]])
        # U'UBVRI sigma
        self._sigma = np.array([0.5900, 0.06001, 0.040034, 0.050014, 0.040017, 0.080007])

    @property
    def covmat(self):
        """Define the covariance matrix according to the choice made for COV_U'U.

        Returns
        -------
        numpy.ndarray(float, size = (6,6))
            Matrice de covariance U'UBVRI.

        Notes
        -----
        cor2cov is multiply by _parameters[1] to rescale the error due to the
        fact taht we pass from broadband to continuous wavelengths.
        """
        covmat = np.zeros((6, 6))
        for i in range(6):
            for j in range(i + 1):
                cor2cov = self._corr_matrix[i, j]
                if i != 0 and j == 0:
                    cor2cov = self._parameters[0] * self._corr_matrix[i, 1]
                sigi_sigj = self._sigma[i] * self._sigma[j]
                cor2cov *= sigi_sigj
                covmat[i, j] = covmat[j, i] = cor2cov * self._parameters[1]
        return covmat

    @property
    def scatter(self):
        """Generate random scatter.

        Returns
        -------
        numpy.ndarray
            The 6 values of random scatter of the SN.
        """
        RS = self._parameters[-1]
        mu = np.zeros(6)
        scat = np.random.default_rng(int(RS)).multivariate_normal(mu,
                                                                  self.covmat,
                                                                  check_valid='raise')
        return scat

    def propagate(self, wave, flux):
        """Propagate the effect to the flux.

        Parameters
        ----------
        wave : float
            wavelength.
        flux : float
            flux density at wavelength.

        Returns
        -------
        numpy.ndarray(float)
            Flux density with effect applied.
        """
        if self._parameters[0] not in [0., 1., -1.]:
            raise ValueError('Cov_uu must be 1,-1 or 0')

        scatter = self.scatter
        scattering = np.zeros(len(wave))
        for i, w in enumerate(wave):
            if w >= self._sigma_lam[-1]:
                scattering[i] = scatter[-1]
            elif w <= self._sigma_lam[0]:
                scattering[i] = scatter[0]
            else:
                scattering[i] = nbf.sine_interp(w, self._sigma_lam, scatter)
        return flux * 10**(-0.4 * scattering)
