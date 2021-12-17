"""This module contain generator class."""
import numpy as np
from . import utils as ut
from . import nb_fun as nbf
from . import dust_utils as dst_ut
from . import scatter as sct
from . import salt_utils as salt_ut


class BaseGen:
    def __init__(self, params, cmb, cosmology, vpec_dist,
                 host=None, alpha_dipole=None):
        self._params = params
        self._cmb = cmb
        self._cosmology = cosmology
        self._host = host
        self._alpha_dipole = alpha_dipole

        self._time_range = None
        self._z_cdf = None
        self.sim_model = None

    def _init_dust(self):
        if 'mw_dust' in self.model_config:
            dst_ut.init_mw_dust(self.sim_model, self.model_config['mw_dust'])

    def gen_transient_par(self, n_sn, rand_gen=None):
        if rand_gen is None:
            rand_gen = np.random.default_rng()

        # -- Generate peak magnitude
        t0 = self.gen_peak_time(n_sn, rand_gen)

        # -- Generate cosmological redshifts
        if self.host is None or self.host.config['distrib'].lower() == 'as_sn':
            zcos = self.gen_zcos(n_sn, rand_gen)
            if self.host is not None:
                treshold = (self.z_cdf[0][-1] - self.z_cdf[0][0]) / 100
                host = self.host.host_near_z(zcos, treshold)
        else:
            host = self.host.random_choice(n_sn, rand_gen)
            zcos = host['redshift'].values

        # -- If there is hosts use them
        if self.host is not None:
            ra = host['ra'].values
            dec = host['dec'].values
            vpec = host['v_radial'].values
        else:
            ra, dec = self.gen_coord(n_sn, np.random.default_rng(opt_seeds[1]))
            vpec = self.gen_vpec(n_sn, np.random.default_rng(opt_seeds[2]))

        # -- Add dust if necessary
        if 'mw_' in self.sim_model.effect_names:
            dust_par = self._dust_par(ra, dec)
        else:
            dust_par = [{}] * len(ra)

        default_par = [('zcos', zcos),
                        ('como_dist', self.cosmology.comoving_distance(zcos).value),
                        ('z2cmb', ut.compute_z2cmb(ra, dec, self.cmb)),
                        ('sim_t0', t0),
                        ('ra', ra),
                        ('dec', dec),
                        ('vpec', vpec)]

        return default_par, dust_par

    @property
    def host(self):
        """Get the host class."""
        return self._host

    @property
    def vpec_dist(self):
        """Get the peculiar velocity distribution parameters."""
        return self._vpec_dist

    @property
    def alpha_dipole(self):
        """Get alpha dipole parameters."""
        return self._alpha_dipole

    @property
    def cosmology(self):
        """Get astropy cosmological model."""
        return self._cosmology

    @property
    def cmb(self):
        """Get cmb used parameters."""
        return self._cmb

    @property
    def time_range(self):
        """Get time range."""
        return self._time_range

    @time_range.setter
    def time_range(self, time_range):
        """Set the time range."""
        if time_range[0] > time_range[1]:
            print('Time range should be [Tmin,Tmax]')
        else:
            self._time_range = time_range

    @property
    def z_cdf(self):
        """Get the redshift cumulative distribution."""
        return self._z_cdf

    @z_cdf.setter
    def z_cdf(self, cdf):
        """Set the redshift cumulative distribution."""
        self._z_cdf = cdf

    @staticmethod
    def _construct_int_par(*arg):
        keys = [a[0] for a in arg]
        data = [a[1] for a in arg]
        dic = ({k: val for k, val in zip(keys, values)} for values in zip(*data))
        return dic

    def gen_peak_time(self, n, rand_gen):
        """Generate uniformly n peak time in the survey time range.

        Parameters
        ----------
        n : int
            Number of time to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float)
            A numpy array which contains generated peak time.

        """
        t0 = rand_gen.uniform(*self.time_range, size=n)
        return t0

    @staticmethod
    def gen_coord(n, rand_gen):
        """Generate n coords (ra,dec) uniformly on the sky sphere.

        Parameters
        ----------
        n : int
            Number of coord to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            2 numpy arrays containing generated coordinates.

        """
        ra = rand_gen.uniform(low=0, high=2 * np.pi, size=n)
        dec_uni = rand_gen.random(size=n)
        dec = np.arcsin(2 * dec_uni - 1)
        return ra, dec

    def gen_zcos(self, n, rand_gen):
        """Generate n cosmological redshift in a range.

        Parameters
        ----------
        n : int
            Number of redshift to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float)
            A numpy array which contains generated cosmological redshift.
        """
        uni_var = rand_gen.random(size=n)
        zcos = np.interp(uni_var, self.z_cdf[1], self.z_cdf[0])
        return zcos

    def gen_vpec(self, n, rand_gen):
        """Generate n peculiar velocities.

        Parameters
        ----------
        n : int
            Number of vpec to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing vpec (km/s) generated.

        """
        vpec = rand_gen.normal(
            loc=self.vpec_dist['mean_vpec'],
            scale=self.vpec_dist['sig_vpec'],
            size=n)
        return vpec

    def _dust_par(self, ra, dec):
        """Compute dust parameters.

        Parameters
        ----------
        ra : numpy.ndaray(float)
            SN Right Ascension.
        dec : numpy.ndarray(float)
            SN Declinaison.

        Returns
        -------
        list(dict)
            List of Dictionnaries that contains Rv and E(B-V) for each SN.

        """
        ebv = dst_ut.compute_ebv(ra, dec)
        if isinstance(self.model_config['mw_dust'], str):
            mod_name = self.model_config['mw_dust']
            r_v = 3.1
        elif isinstance(self.model_config['mw_dust'], (list, np.ndarray)):
            mod_name = self.model_config['mw_dust'][0]
            r_v = self.model_config['mw_dust'][1]
        if mod_name.lower() in ['ccm89', 'od94']:
            r_v = np.ones(len(ra)) * r_v
            dust_par = [{'mw_r_v': r, 'mw_ebv': e} for r, e in zip(r_v, ebv)]
        elif mod_name.lower() in ['f99']:
            dust_par = [{'mw_ebv': e} for e in ebv]
        else:
            raise ValueError(f'{mod_name} is not implemented')
        return dust_par


class SNIaGen(BaseGen):
    _available_models = ['salt2', 'salt3']

    def __init__(self, params, cmb, cosmology, vpec_dist,
                 host=None):
        super().__init__(cmb, cosmology, vpec_dist,
                         host=host, alpha_dipole=alpha_dipole)
        self.sim_model = self._init_sim_model()
        self._init_dust()
        self._general_par = self._init_general_par()

    def __call__(self, n_sn, rand_gen=None):
        # Generate default transient parameters
        sn_int_par, dust_par = super().gen_transient_par(n_sn, rand_gen=rand_gen)

        # -- Generate coherent mag scattering
        sn_int_par.append(('mag_sct', self.gen_coh_scatter(n_sn, rand_gen)))

        if 'dipole' in self._params:
            sn_int_par.append(('adip_dM', self._compute_dipole(ra, dec)))

        rand_model_par = self.gen_model_par(n_sn, np.random.default_rng(opt_seeds[0]), z=zcos)

        # Convert sn_int_par to dict
        sn_int_par_dict = self._construct_sn_int(*sn_int_par)

        model_par_list = ({**self.general_par, 'sncosmo': {**mpsn, **dstp}}
                          for mpsn, dstp in zip(rand_model_par, dust_par))

        return [SNIa(snp, self.sim_model, mp) for snp, mp in zip(sn_int_par_dict, model_par_list)]

    def _init_M0(self):
        if isinstance(self._params['M0'], (float, int)):
            return self._params['M0']

        elif self._params['M0'].lower() == 'jla':
             return ut.scale_M0_jla(self.cosmology.H0.value)

    def _init_sim_model(self):
        """Initialise sncosmo model using the good source.

        Returns
        -------
        sncosmo.Model
            sncosmo.Model(source) object where source depends on the
            SN simulation model.

        """
        if self._params['model_config']['model_name'].lower() not in self._available_models:
            raise ValueError('Model not available')

        model_dir = None
        if 'model_dir' in self._params['model_config']:
            model_dir = self._params['model_config']['model_dir']

        model = ut.init_sn_model(self._params['model_config']['model_name'],
                                 model_dir)

        if 'sct_model' in self._params:
            sct.init_sn_sct_model(model, self._params['sct_model'])
        return model

    def _init_general_par(self):
        """Initialise the general parameters, depends on the SN simulation model.

        Returns
        -------
        list
            A dict containing all the usefull keys of the SN model.
        """
        model_name = self._params['model_config']['model_name']
        if model_name in ('salt2', 'salt3'):
            model_keys = ['alpha', 'beta']

        model_par = {'M0': self._init_M0()}
        for k in model_keys:
            model_par[k] = self._params['model_config'][k]

        if 'mod_fcov' not in self._params['model_config']:
            model_par['mod_fcov'] = False
        else:
            model_par['mod_fcov'] = self._params['model_config']['mod_fcov']

        return model_par

    def gen_coh_scatter(self, n, rand_gen):
        """Generate n coherent mag scattering term.

        Parameters
        ----------
        n : int
            Number of mag scattering terms to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float)
            numpy array containing scattering terms generated.

        """
        mag_sct = rand_gen.normal(
            loc=0, scale=self.sn_int_par['mag_sct'], size=n)
        return mag_sct

    def gen_model_par(self, n, rand_gen, z=None):
        """Generate model dependant parameters.

        Parameters
        ----------
        n : int
            Number of parameters to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        dict
            One dictionnary containing 'parameters names': numpy.ndaray(float).

        """
        model_name = self.model_config['model_name']

        if model_name in ('salt2', 'salt3'):

            if self.model_config['dist_x1'] in ['N21']:
                z_for_dist = z
            else:
                z_for_dist = None

            sim_x1, sim_c = self.gen_salt_par(n, rand_gen, z=z_for_dist)
            model_par_sncosmo = [{'x1': x1, 'c': c} for x1, c in zip(sim_x1, sim_c)]

        if 'G10_' in self.sim_model.effect_names:
            seeds = rand_gen.integers(low=1000, high=100000, size=n)
            for par, s in zip(model_par_sncosmo, seeds):
                par['G10_RndS'] = s

        elif 'C11_' in self.sim_model.effect_names:
            seeds = rand_gen.integers(low=1000, high=100000, size=n)
            for par, s in zip(model_par_sncosmo, seeds):
                par['C11_RndS'] = s

        return model_par_sncosmo

    def gen_salt_par(self, n, rand_gen, z=None):
        """Generate n SALT parameters.

        Parameters
        ----------
        n : int
            Number of parameters to generate.
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        numpy.ndarray(float), numpy.ndarray(float)
            2 numpy arrays containing SALT2 x1 and c generated parameters.

        """
        if isinstance(self.model_config['dist_x1'], str):
            if self.model_config['dist_x1'].lower() == 'n21':
                sim_x1 = salt_ut.n21_x1_model(z, rand_gen=rand_gen)
        else:
            sim_x1 = ut.asym_gauss(*self.model_config['dist_x1'],
                                   rand_gen=rand_gen,
                                   size=n)

        sim_c = ut.asym_gauss(*self.model_config['dist_c'],
                              rand_gen=rand_gen,
                              size=n)

        return sim_x1, sim_c

    def _compute_dipole(self, ra, dec):
        cart_vec = nbf.radec_to_cart(self._params['dipole']['coord'][0],
                                     self._params['dipole']['coord'][1])
        sn_vec = ut.radec_to_cart(ra, dec)
        delta_M =self._params['dipole']['A'] + self._params['dipole']['B'] * cart_vec @ sn_vec)
        return delta_M
