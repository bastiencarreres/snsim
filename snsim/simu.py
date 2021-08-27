"""Main module of the simulaiton package."""

import pickle
import time
import yaml
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from . import utils as ut
from . import sim_class as scls
from . import plot_utils as plot_ut
from .constants import SN_SIM_PRINT, VCMB, L_CMB, B_CMB
from . import dust_utils as dst_ut


class Simulator:
    """Simulation class using a config file config.yml.

    Parameters
    ----------
    param_dic : dict / str
        The configuration yaml file /
        The dictionnary containing all simulation parameters.

    Attributes
    ----------
    _config : dict
        The simulation parameters dictionnary
    _sn_list : list (init value = None)
        List containing simulated SN object
    _fit_res : list (init value = None)
        List containing sncosmo fit results
    _random_seed : int (init value = None)
        The primary random seed of the simulation
    _host : SnHost (default = None)
        Host object of the simulation
    _obs : ObsTable
        ObsTable object of the simulation.
    _generator : SNGen
        SNGen object of the simulation
    _use_rate : boolean
        Is the number of SN to simulate fixed?

    Methods
    -------
    sn_rate(z)
        Give the rate SNs/Mpc^3/year at redshift z.
    simulate()
        Launch the simulation.
    plot_lc(sn_ID, mag = False, zp = 25., plot_sim = True, plot_fit = False)
        Plot the given SN lightcurve.
    plot_ra_dec(self, plot_vpec=False, **kwarg):
        Plot a mollweide map of ra, dec.
    fit_lc(sn_ID = None)
        Fit all or just one SN lightcurve(s).
    write_fit()
        Write fits results in fits format.
    _time_rate_bins()
        Give the time rate SN/years in redshift bins.
    _gen_n_sn(rand_seed)
        Generate the number of SN with Poisson law.
    _cadence_sim()
        Simulaton where the number of SN observed is determined by
        survey properties and poisson law.
    _fix_nsn_sim()
        Simulation where the number of SN is fixed.
    _get_primary_header()
        Generate the primary header of sim fits file.
    _write_sim()
        Write sim lightcurves in fits or/and pkl format(s).

    Notes
    -----
    - If the name of bands in the survey file doesn't match sncosmo bands
    you can use the key band_dic to translate filters names
    - If you don't set the filter name item in nep_cut, the cut apply to all the bands
    - For wavelength dependent model, nomanclature follow arXiv:1209.2482
        Possibilities are :
        -> 'G10' for Guy et al. 2010 model,
        -> 'C11' or 'C11_0' for Chotard et al. model with correlation between
        U' and U = 0, 'C11_1' for Cor(U',U) = 1 and 'C11_2' for Cor(U',U) = -1

    yaml file format :

    +------------------------------------------------------------------------------------+
    | data :                                                                             |
    |     write_path: '/PATH/TO/OUTPUT'                                                  |
    |     sim_name: 'NAME OF SIMULATION'                                                 |
    |     write_format: 'format' or ['format1','format2']                                |
    | survey_config:                                                                     |
    |     survey_file: '/PATH/TO/FILE'                                                   |
    |     band_dic: {'r':'ztfr','g':'ztfg','i':'ztfi'} #(Optional -> if bandname in      |
    |  survey_file doesn't match sncosmo name)                                           |
    |     add_data: ['keys1', 'keys2', ...] add db file keys to metadata                 |
    |     survey_cut: {'key1': ['conditon1','conditon2',...], 'key2': ['conditon1']}     |
    |     start_day: MJD NUMBER or 'YYYY-MM-DD'(Optional, default given by survey file)  |
    |     duration: SURVEY DURATION (DAYS) (Optional, default given by survey file)      |
    |     zp: INSTRUMENTAL ZEROPOINT (Optional, default given by survey file)            |
    |     sig_zp: ZEROPOINT ERROR (Optional, default given by survey file)               |
    |     sig_psf: GAUSSIAN PSF SIGMA (Otional, default given by survey file as FWHMeff) |
    |     noise_key: [key, type] type can be 'mlim5' or 'skysigADU'                      |
    |     ra_size: RA FIELD SIZE                                                         |
    |     dec_size: DEC FIELD SIZE                                                       |
    |     gain: CCD GAIN e-/ADU                                                          |
    |     sub_field: ['sub_field_file', 'sub_field_key']                                 |
    | sn_gen:                                                                            |
    |     n_sn: NUMBER OF SN TO GENERATE (Otional)                                       |
    |     duration_for_rate: FAKE DURATION ONLY USE TO GENERATE N SN (Optional)          |
    |     sn_rate: rate of SN/Mpc^3/year (Optional, default=3e-5)                        |
    |     rate_pw: rate = sn_rate*(1+z)^rate_pw (Optional, default=0)                    |
    |     nep_cut: [[nep_min1,Tmin,Tmax],[nep_min2,Tmin2,Tmax2,'filter1'],...] EP CUTS   |
    |     randseed: RANDSEED TO REPRODUCE SIMULATION #(Optional)                         |
    |     z_range: [ZMIN, ZMAX]                                                          |
    |     M0: SN ABSOLUT MAGNITUDE                                                       |
    |     mag_sct: SN INTRINSIC COHERENT SCATTERING                                      |
    |     sct_model: 'G10','C11_i' USE WAVELENGHT DEP MODEL FOR SN INT SCATTERING        |
    | cosmology:                                                                         |
    |     Om0: MATTER DENSITY                                                            |
    |     H0: HUBBLE CONSTANT                                                            |
    | cmb:                                                                               |
    |     v_cmb: OUR PECULIAR VELOCITY #(Optional, default = 620 km/s)                   |
    |     l_cmb: GAL L OF CMB DIPOLE #(Optional, default = 271.0)                        |
    |     b_cmb: GAL B OF CMB DIPOLE #(Optional, default = 29.6)                         |
    | model_config:                                                                      |
    |     model_name: 'THE MODEL NAME'  Example : 'salt2'                                |
    |     model_dir: '/PATH/TO/SALT/MODEL'                                               |
    |     alpha: STRETCH CORRECTION = alpha*x1                                           |
    |     beta: COLOR CORRECTION = -beta*c                                               |
    |     mean_x1: MEAN X1 VALUE                                                         |
    |     mean_c: MEAN C VALUE                                                           |
    |     sig_x1: SIGMA X1 or [SIGMA_X1_LOW, SIGMA_X1_HIGH]                              |
    |     sig_c: SIGMA C or [SIGMA_C_LOW, SIGMA_C_HIGH]                                  |
    |     mw_dust: MOD_NAME #(RV = 3.1) or [MOD_NAME, RV]  #(Optional)                   |
    | vpec_dist:                                                                         |
    |     mean_vpec: MEAN SN PECULIAR VELOCITY                                           |
    |     sig_vpec: SIGMA VPEC                                                           |
    | host_file: 'PATH/TO/HOSTFILE' (Optional)                                           |
    | alpha_dipole: #Experimental alpha fine structure constant dipole, optional         |
    |     coord: [RA, Dec] # Direction of the dipole                                     |
    |     A: A_parameter # alpha dipole = A + B * cos(theta)                             |
    |     B: B_parameter                                                                 |
    |                                                                                    |
    +------------------------------------------------------------------------------------+
    """

    def __init__(self, param_dic):
        """Initialise Simulator class."""
        # Load param dict from a yaml or by using launch_script.py
        if isinstance(param_dic, dict):
            self._config = param_dic
            if 'yaml_path' in param_dic:
                self._yml_path = param_dic['yaml_path']
            else:
                self._yml_path = 'No config file'

        elif isinstance(param_dic, str):
            self._yml_path = param_dic
            with open(self._yml_path, "r") as f:
                self._config = yaml.safe_load(f)

        # Check if there is a db_file
        if 'survey_config' not in self.config:
            raise KeyError("Set a survey_file -> type help(sn_sim) to print the syntax")

        # Check if the sfdmap need to be download
        if 'mw_dust' in self.config['model_config']:
            dst_ut.check_files_and_dowload()

        # cadence sim or n fixed
        if 'n_sn' in self.config['sn_gen']:
            self._use_rate = False
        else:
            self._use_rate = True

        self._sn_list = None
        self._fit_res = None
        self._fit_resmod = None
        self._random_seed = None
        self._host = None

        self._cosmology = FlatLambdaCDM(**self.config['cosmology'])
        self._survey = scls.SurveyObs(self.config['survey_config'])
        self._generator = scls.SnGen(self.sn_int_par,
                                     self.config['model_config'],
                                     self.cmb,
                                     self.cosmology,
                                     self.vpec_dist,
                                     host=self.host,
                                     alpha_dipole=self.alpha_dipole)

    @property
    def config(self):
        """Get the whole configuration dic."""
        return self._config

    @property
    def sim_name(self):
        """Get sim name."""
        return self.config['data']['sim_name']

    @property
    def vpec_dist(self):
        """Get vpec option."""
        if 'vpec_dist' in self.config:
            return self.config['vpec_dist']
        elif 'host_file' in self.config:
            return None
        else:
            return {'mean_vpec': 0., 'sig_vpec': 0.}

    @property
    def cmb(self):
        """Get cmb parameters."""
        if 'cmb' in self.config:
            if 'v_cmb' in self.config['cmb']:
                v_cmb = self.config['cmb']['v_cmb']
            else:
                v_cmb = VCMB
            if 'l_cmb' in self.config['cmb']:
                l_cmb = self.config['cmb']['l_cmb']
            else:
                l_cmb = L_CMB
            if 'b_cmb' in self.config['cmb']:
                b_cmb = self.config['cmb']['b_cmb']
            else:
                b_cmb = B_CMB
        else:
            v_cmb = VCMB
            l_cmb = L_CMB
            b_cmb = B_CMB
        return {'v_cmb': v_cmb, 'l_cmb': l_cmb, 'b_cmb': b_cmb}

    @property
    def n_sn(self):
        """Get number of sn simulated."""
        if self._sn_list is None:
            print('You have to run the simulation')
            return None
        return len(self._sn_list)

    @property
    def model_name(self):
        """Get the name of sn model used."""
        return self.config['model_config']['model_name']

    @property
    def sn_list(self):
        """Get the list of simulated sn."""
        return self._sn_list

    @property
    def fit_res(self):
        """Get the sn fit results."""
        return self._fit_res

    @property
    def fit_resmod(self):
        """Get the sn fit results sncosmo models."""
        return self._fit_resmod

    @property
    def sn_int_par(self):
        """Get the intrinsic parameters of SN Ia."""
        int_params = {'M0': self.config['sn_gen']['M0'],
                      'mag_sct': self.config['sn_gen']['mag_sct']}
        if 'sct_model' in self.config['sn_gen']:
            int_params['sct_model'] = self.config['sn_gen']['sct_model']
        return int_params

    @property
    def cosmology(self):
        """Get astropy cosmological model used in simulation."""
        return self._cosmology

    @property
    def survey(self):
        """Get the SurveyObs object of the simulation."""
        return self._survey

    @property
    def generator(self):
        """Get the SNGen object of the simulation."""
        return self._generator

    @property
    def host(self):
        """Get the SnHost object of the simulation."""
        if self._host is not None:
            return self._host
        elif 'host_file' in self.config:
            self._host = scls.SnHost(self.config['host_file'], self.z_range)
            return self._host
        else:
            return None

    @property
    def nep_cut(self):
        """Get the list of epochs cuts."""
        snc_mintime, snc_maxtime = self.generator.snc_model_time
        if 'nep_cut' in self.config['sn_gen']:
            nep_cut = self.config['sn_gen']['nep_cut']
            if isinstance(nep_cut, (int)):
                nep_cut = [
                    (nep_cut,
                     snc_mintime,
                     snc_maxtime)]
            elif isinstance(nep_cut, (list)):
                for i, cut in enumerate(nep_cut):
                    if len(cut) < 3:
                        nep_cut[i].append(snc_mintime)
                        nep_cut[i].append(snc_maxtime)
        else:
            nep_cut = [(1, snc_mintime, snc_maxtime)]
        return nep_cut

    @property
    def rand_seed(self):
        """Get primary random seed of the simulation."""
        if 'randseed' in self.config['sn_gen']:
            return int(self.config['sn_gen']['randseed'])
        elif self._random_seed is None:
            self._random_seed = np.random.randint(low=1000, high=100000)
        return self._random_seed

    @property
    def z_range(self):
        """Get the simulation cosmological redshift range."""
        return self.config['sn_gen']['z_range']

    @property
    def sn_rate_z0(self):
        """Get the sn rate parameters."""
        if 'sn_rate' in self.config['sn_gen']:
            sn_rate = float(self.config['sn_gen']['sn_rate'])
        else:
            sn_rate = 3e-5
        if 'rate_pw' in self.config['sn_gen']:
            rate_pw = self.config['sn_gen']['rate_pw']
        else:
            rate_pw = 0
        return sn_rate, rate_pw

    @property
    def peak_time_range(self):
        """Get the time range for simulate SN peak.

        Returns
        -------
        tuple(float, float)
            Min and max time for SN peak generation.
        """
        min_peak_time = self.survey.start_end_days[0] - self.generator.snc_model_time[1] \
            * (1 + self.z_range[1])
        max_peak_time = self.survey.start_end_days[1] + abs(self.generator.snc_model_time[0]) \
            * (1 + self.z_range[1])
        return min_peak_time, max_peak_time

    @property
    def alpha_dipole(self):
        """Get alpha dipole parameters."""
        if 'alpha_dipole' in self.config:
            return self.config['alpha_dipole']
        return None

    def sn_rate(self, z):
        """Give the rate SNs/Mpc^3/year at redshift z.

        Parameters
        ----------
        z : float, numpy.ndarray(float)
            One of a list of cosmological redshift(s).

        Returns
        -------
        float, numpy.ndarray(float)
            One or a list of sn rate(s) corresponding to input redshift(s).

        """
        rate_z0, rpw = self.sn_rate_z0
        return rate_z0 * (1 + z)**rpw

    def _z_shell_time_rate(self):
        """Give the time rate SN/years in redshift shell.

        Parameters
        ----------
        z : numpy.ndarray
            The redshift bins.

        Returns
        -------
        numpy.ndarray(float)
            Numpy array containing the time rate in each redshift bin.

        """
        z_min, z_max = self.z_range
        z_shell = np.linspace(z_min, z_max, 1000)
        z_shell_center = 0.5 * (z_shell[1:] + z_shell[:-1])
        rate = self.sn_rate(z_shell_center)  # Rate in Nsn/Mpc^3/year
        co_dist = self.cosmology.comoving_distance(z_shell).value
        shell_vol = 4 * np.pi / 3 * (co_dist[1:]**3 - co_dist[:-1]**3)

        # -- Compute the sn time rate in each volume shell [( SN / year )(z)]
        shell_time_rate = rate * shell_vol / (1 + z_shell_center)
        return z_shell, shell_time_rate

    def _gen_n_sn(self, rand_gen, z_shell_time_rate):
        """Generate the number of SN with Poisson law.

        Parameters
        ----------
        rand_seed : int
            The random seed for the random generator.

        Returns
        -------
        numpy.ndarray
            Numpy array containing the simulated number of SN in each redshift
            bin.

        """
        if 'duration_for_rate' in self.config['sn_gen']:
            time_in_days = self.config['sn_gen']['duration_for_rate']
        else:
            min_peak_time, max_peak_time = self.peak_time_range
            time_in_days = max_peak_time.mjd - min_peak_time.mjd
        return rand_gen.poisson(time_in_days / 365.25 * np.sum(z_shell_time_rate))

    def simulate(self):
        """Launch the simulation.

        Returns
        -------
        None

        Notes
        -----
        Simulation routine :
        1- Use either _cadence_sim() or _gen_n_sn()
        to run the simulation
        2- Gen all SN parameters inside SNGen class or/and SnHost class
        3- Check if SN pass cuts and then generate the lightcurves.
        4- Write LC to fits/pkl file(s)

        """
        print(SN_SIM_PRINT)
        print('-----------------------------------------------------------')
        print(f"SIM NAME : {self.sim_name}\n"
              f"CONFIG FILE : {self._yml_path}\n"
              f"SURVEY FILE : {self.config['survey_config']['survey_file']}")
        if 'host_file' in self.config:
            print(f"HOST FILE : {self.config['host_file']}")
        print(f"SN SIM MODEL : {self.model_name} from {self.config['model_config']['model_dir']}\n"
              f"SIM WRITE DIRECTORY : {self.config['data']['write_path']}\n"
              f"SIMULATION RANDSEED : {self.rand_seed}")

        print('-----------------------------------------------------------')

        if self._use_rate:
            use_rate_str = ''
        else:
            print(f"Generate {self.config['sn_gen']['n_sn']} SN Ia\n")
            use_rate_str = ' (only for redshifts simulation)'

        print(f"SN rate of r_v = {self.sn_rate_z0[0]}*(1+z)^{self.sn_rate_z0[1]} SN/Mpc^3/year"
              + use_rate_str + "\n"
              "SN peak mintime : "
              f"{self.peak_time_range[0].mjd:.2f} MJD / {self.peak_time_range[0].iso}\n"
              "SN peak maxtime : "
              f"{self.peak_time_range[1].mjd:.2f} MJD / {self.peak_time_range[1].iso} \n\n"
              "First day in survey_file : "
              f"{self.survey.start_end_days[0].mjd:.2f} MJD / {self.survey.start_end_days[0].iso}\n"
              "Last day in survey_file : "
              f"{self.survey.start_end_days[1].mjd:.2f} MJD / {self.survey.start_end_days[1].iso}")

        if 'duration_for_rate' in self.config['sn_gen']:
            print(
                "N SN is generate for a duration of "
                f"{self.config['sn_gen']['duration_for_rate']:.2f} days")
        else:
            print(f"Survey effective duration is {self.survey.duration:.2f} days")

        if 'sct_model' in self.config['sn_gen']:
            print("\nUse intrinsic scattering model : "
                  f"{self.config['sn_gen']['sct_model']}")

        if 'mw_dust' in self.config['model_config']:
            print("\nUse mw dust model : "
                  f"{np.atleast_1d(self.config['model_config']['mw_dust'])[0]}")

        print('-----------------------------------------------------------\n')

        if 'survey_cut' in self.config['survey_config']:
            for k, v in self.config['survey_config']['survey_cut'].items():
                conditions_str = ''
                for cond in v:
                    conditions_str += str(cond) + ' OR '
                conditions_str = conditions_str[:-4]
                print(f'Select {k}: ' + conditions_str)
        else:
            print('No db cut')

        print('\n-----------------------------------------------------------\n')

        print("SN ligthcurve cuts :")

        for cut in self.nep_cut:
            print_cut = f'- At least {cut[0]} epochs between {cut[1]} and {cut[2]}'
            if len(cut) == 4:
                print_cut += f' in {cut[3]} band'
            print(print_cut)

        print('\n-----------------------------------------------------------\n')

        sim_time = time.time()

        # -- Init the redshift distribution
        z_shell, shell_time_rate = self._z_shell_time_rate()
        self.generator.z_cdf = ut.compute_z_cdf(z_shell, shell_time_rate)

        # -- Set the time range with time edges effects
        self.generator.time_range = [self.peak_time_range[0].mjd, self.peak_time_range[1].mjd]

        # -- Init the sn list
        self._sn_list = []

        # -- Create the random generator object with the rand seed
        rand_gen = np.random.default_rng(self.rand_seed)

        if self._use_rate:
            self._cadence_sim(rand_gen, shell_time_rate)
        else:
            self._fix_nsn_sim(rand_gen)

        print(f'{len(self._sn_list)} SN lcs generated in {time.time() - sim_time:.1f} seconds')

        print('\n-----------------------------------------------------------\n')

        write_time = time.time()
        self._write_sim()

        print(f'Sim file write in {time.time() - write_time:.1f} seconds')

        print('\n-----------------------------------------------------------\n')

        print('OUTPUT FILE(S) : ')
        if isinstance(self.config['data']['write_format'], str):
            print(self.config['data']['write_path']
                  + self.sim_name
                  + '.'
                  + self.config['data']['write_format'])
        else:
            for f in self.config['data']['write_format']:
                print('- '
                      + self.config['data']['write_path']
                      + self.sim_name
                      + '.'
                      + f)
        print("\n")

    def _cadence_sim(self, rand_gen, shell_time_rate):
        """Simulate a number of SN according to poisson law.

        Parameters
        ----------
        rand_gen : numpy.random.default_rng
            Numpy random generator.
        shell_time_rate : numpy.ndarray
            An array that contains sn time rate in each shell.

        Returns
        -------
        None

        Notes
        -----
        Simulation routine:
            1- Cut the zrange into shell (z,z+dz)
            2- Compute time rate for the shell r = r_v(z) * V SN/year where r_v is the volume rate
            3- Generate the number of SN Ia in each shell with a Poisson's law
            4- Generate ra, dec for all the SN uniform on the sphere
            5- Generate t0 uniform between mintime and maxtime
            5- Generate z for in each shell uniform in the interval [z,z+dz]
            6- Apply observation and selection cuts to SN

        """
        # -- Generate the number of SN
        n_sn = self._gen_n_sn(rand_gen, shell_time_rate)

        SN_ID = 0
        sn_list_tmp = self.generator(n_sn, rand_gen)
        for sn in sn_list_tmp:
            sn.epochs = self.survey.epochs_selection(sn)
            if sn.pass_cut(self.nep_cut):
                sn.gen_flux(rand_gen)
                sn.ID = SN_ID
                SN_ID += 1
                self._sn_list.append(sn)

    def _fix_nsn_sim(self, rand_gen):
        """Simulate a fixed number of SN.

        Parameters
        ----------
        rand_gen : numpy.random.default_rng
            Numpy random generator.

        Returns
        -------
        None

        Notes
        -----
        Just generate SN randomly until we reach the desired number of SN.

        """
        raise_trigger = 0
        SN_ID = 0
        while len(self._sn_list) < self.config['sn_gen']['n_sn']:
            sn = self.generator(1, rand_gen)[0]
            sn.epochs = self.survey.epochs_selection(sn)
            if sn.pass_cut(self.nep_cut):
                sn.gen_flux(rand_gen)
                sn.ID = SN_ID
                SN_ID += 1
                self._sn_list.append(sn)
            elif raise_trigger > 2 * len(self.survey.obs_table['expMJD']):
                print(len(self.survey.obs_table['expMJD']))
                raise RuntimeError('Cuts are too stricts')
            else:
                raise_trigger += 1

    def _get_primary_header(self):
        """Generate the primary header of sim fits file..

        Returns
        -------
        None

        """
        header = {'n_sn': self.n_sn,
                  'M0': self.config['sn_gen']['M0'],
                  'sigM': self.config['sn_gen']['mag_sct'],
                  **self.config['cosmology']}

        if self.host is None:
            header['m_vp'] = self.config['vpec_dist']['mean_vpec']
            header['s_vp'] = self.config['vpec_dist']['sig_vpec']

        if self.model_name == 'salt2' or self.model_name == 'salt3':
            fits_dic = {'model_name': 'Mname',
                        'alpha': 'alpha',
                        'beta': 'beta',
                        'mean_x1': 'm_x1',
                        'mean_c': 'm_c'}

            if isinstance(self.config['model_config']['sig_x1'], list):
                header['s_x1_sup'] = self.config['model_config']['sig_x1'][0]
                header['s_x1_inf'] = self.config['model_config']['sig_x1'][1]
            else:
                fits_dic['sig_x1'] = 's_x1'

            if isinstance(self.config['model_config']['sig_c'], list):
                header['s_c_sup'] = self.config['model_config']['sig_c'][0]
                header['s_c_inf'] = self.config['model_config']['sig_c'][1]
            else:
                fits_dic['sig_c'] = 's_c'

        if 'mw_dust' in self.config['model_config']:
            if isinstance(self.config['model_config']['mw_dust'], (list, np.ndarray)):
                header['mwd_mod'] = self.config['model_config']['mw_dust'][0]
                header['mw_rv'] = self.config['model_config']['mw_dust'][1]
            else:
                header['mwd_mod'] = self.config['model_config']['mw_dust']
                header['mw_rv'] = 3.1

        for k, v in fits_dic.items():
            header[v] = self.config['model_config'][k]

        if 'sct_model' in self.config['sn_gen']:
            header['Smod'] = self.config['sn_gen']['sct_model']
        return header

    def _write_sim(self):
        """Write sim lightcurves in fits or/and pkl format(s).

        Returns
        -------
        None
            Just write sim into a file

        """
        write_path = self.config['data']['write_path']
        sim_header = self._get_primary_header()
        if 'fits' in self.config['data']['write_format']:
            lc_hdu_list = (sn.get_lc_hdu() for sn in self._sn_list)
            hdu_list = fits.HDUList(
                [fits.PrimaryHDU(header=fits.Header(sim_header))] + list(lc_hdu_list))

            hdu_list.writeto(write_path + self.sim_name + '.fits',
                             overwrite=True)

        # Export lcs as pickle
        if 'pkl' in self.config['data']['write_format']:
            sim_lc = [sn.sim_lc for sn in self._sn_list]
            sn_pkl = scls.SnSimPkl(sim_lc, sim_header)
            with open(write_path + self.sim_name + '.pkl', 'wb') as file:
                pickle.dump(sn_pkl, file)

    def plot_lc(self, sn_ID, mag=False, zp=25., plot_sim=True, plot_fit=False, Jy=False):
        """Plot the given SN lightcurve.

        Parameters
        ----------
        sn_ID : int
            The Supernovae ID.
        mag : boolean, default = False
            If True plot the magnitude instead of the flux.
        zp : float
            Used zeropoint for the plot.
        plot_sim : boolean, default = True
            If True plot the theorical simulated lightcurve.
        plot_fit : boolean, default = False
            If True plot the fitted lightcurve.
        Jy : boolean, default = False
            If True plot in Jansky

        Returns
        -------
        None
            Just plot the SN lightcurve !

        Notes
        -----
        Use plot_lc from utils.

        """
        sn = self._sn_list[sn_ID]
        if plot_sim:
            s_model = self.generator.sim_model.__copy__()
            dic_par = {**{'z': sn.z, 't0': sn.sim_t0}, **sn._model_par['sncosmo']}
            s_model.set(**dic_par)
        else:
            s_model = None

        if plot_fit:
            if self.fit_res is None or self.fit_res[sn_ID] is None:
                print('This SN was not fitted, launch fit')
                if 'mw_dust' in self.config['model_config']:
                    print('Use sim input mw dust')
                    mw_dust = self.config['model_config']['mw_dust']
                else:
                    mw_dust = None
                self.fit_lc(sn_ID, mw_dust=mw_dust)

            if self.fit_res[sn_ID] == 'NaN':
                print('This sn cannot be fitted')
                return
            f_model = self.fit_resmod[sn_ID]
            cov_t0_x0_x1_c = self.fit_res[sn_ID]['covariance'][:, :]
            residuals = True
        else:
            f_model = None
            cov_t0_x0_x1_c = None
            residuals = False

        plot_ut.plot_lc(sn.sim_lc, mag=mag,
                        snc_sim_model=s_model,
                        snc_fit_model=f_model,
                        fit_cov=cov_t0_x0_x1_c,
                        zp=zp,
                        residuals=residuals,
                        Jy=Jy)

    def get_sn_par(self, key):
        """Get an array of a sim_lc meta data."""
        return np.array([sn.sim_lc.meta[key] for sn in self.sn_list])

    def plot_ra_dec(self, plot_vpec=False, plot_fields=False, **kwarg):
        """Plot a mollweide map of ra, dec.

        Parameters
        ----------
        plot_vpec : boolean
            If True plot a vpec colormap.

        Returns
        -------
        None
            Just plot the map.

        """
        ra = []
        dec = []
        vpec = None
        if plot_vpec:
            vpec = []
        if plot_fields:
            field_list = []

        for sn in self.sn_list:
            r, d = sn.coord
            ra.append(r)
            dec.append(d)
            if plot_vpec:
                vpec.append(sn.vpec)
            if plot_fields:
                field_list = np.concatenate((field_list, np.unique(sn.sim_lc['fieldID'])))

        if plot_fields:
            field_dic = self.survey.fields._dic
            field_size = self.survey.fields.size
            field_list = np.unique(field_list)
        else:
            field_dic = None
            field_size = None
            field_list = None

        plot_ut.plot_ra_dec(np.asarray(ra),
                            np.asarray(dec),
                            vpec,
                            field_list,
                            field_dic,
                            field_size,
                            **kwarg)

    def fit_lc(self, sn_ID=None, mw_dust=-2):
        """Fit all or just one SN lightcurve(s).

        Parameters
        ----------
        sn_ID : int, default is None
            The SN ID, if not specified all SN are fit.

        Returns
        -------
        None
            Directly modified the _fit_res attribute.

        Notes
        -----
        Use snc_fitter from utils

        """
        if self.sn_list is None:
            print('No sn to fit, run the simulation before')
            return

        if self._fit_res is None:
            self._fit_res = [None] * len(self.sn_list)
            self._fit_resmod = [None] * len(self.sn_list)

        fit_model = ut.init_sn_model(self.model_name, self.config['model_config']['model_dir'])

        mw_mod = None
        if mw_dust == -2 and 'mw_dust' in self.config['model_config']:
            mw_mod = self.config['model_config']['mw_dust']
        elif isinstance(mw_dust, (str, list, np.ndarray)):
            mw_mod = mw_dust
        else:
            print('Do not use mw dust')

        if mw_mod is not None:
            dst_ut.init_mw_dust(fit_model, mw_mod)
            if isinstance(mw_mod, (list, np.ndarray)):
                rv = mw_mod[1]
                print_mod = mw_mod[0]
            else:
                rv = 3.1
                print_mod = mw_mod
            print(f'Use MW dust model {print_mod} with RV = {rv}')

        if self.model_name in ('salt2', 'salt3'):
            fit_par = ['t0', 'x0', 'x1', 'c']

        if sn_ID is None:
            for i, sn in enumerate(self.sn_list):
                if self._fit_res[i] is None:
                    fit_model.set(z=sn.z)
                    if mw_dust is not None:
                        dst_ut.add_mw_to_fit(fit_model, sn.mw_ebv, rv=rv)
                    self._fit_res[i], self._fit_resmod[i] = ut.snc_fitter(sn.sim_lc,
                                                                          fit_model,
                                                                          fit_par)
        else:
            fit_model.set(z=self.sn_list[sn_ID].z)
            if mw_dust is not None:
                dst_ut.add_mw_to_fit(fit_model, self.sn_list[sn_ID].mw_ebv, rv=rv)
            self._fit_res[sn_ID], self._fit_resmod[sn_ID] = ut.snc_fitter(
                self.sn_list[sn_ID].sim_lc,
                fit_model,
                fit_par)

    def write_fit(self):
        """Write fits results in fits format.

        Returns
        -------
        None
            Write an output file.

        Notes
        -----
        Use write_fit from utils.

        """
        sim_lc_meta = {'sn_id': [sn.ID for sn in self.sn_list],
                       'ra': [sn.coord[0] for sn in self.sn_list],
                       'dec': [sn.coord[1] for sn in self.sn_list],
                       'vpec': [sn.vpec for sn in self.sn_list],
                       'zpec': [sn.zpec for sn in self.sn_list],
                       'z2cmb': [sn.z2cmb for sn in self.sn_list],
                       'zcos': [sn.zcos for sn in self.sn_list],
                       'zCMB': [sn.zCMB for sn in self.sn_list],
                       'zobs': [sn.z for sn in self.sn_list],
                       'sim_mu': [sn.sim_mu for sn in self.sn_list],
                       'sim_t0': [sn.sim_t0 for sn in self.sn_list]}

        if self.model_name == 'salt2' or self.model_name == 'salt3':
            sim_lc_meta['sim_mb'] = [sn.sim_mb for sn in self.sn_list]
            sim_lc_meta['sim_x1'] = [sn.sim_x1 for sn in self.sn_list]
            sim_lc_meta['sim_c'] = [sn.sim_c for sn in self.sn_list]
            sim_lc_meta['m_sct'] = [sn.mag_sct for sn in self.sn_list]

        if self.model_name in ('salt2', 'salt3'):
            sim_lc_meta['sim_x0'] = [sn.sim_x0 for sn in self.sn_list]
            sim_lc_meta['sim_mb'] = [sn.sim_mb for sn in self.sn_list]
            sim_lc_meta['sim_x1'] = [sn.sim_x1 for sn in self.sn_list]
            sim_lc_meta['sim_c'] = [sn.sim_c for sn in self.sn_list]

        if 'sct_model' in self.config['sn_gen']:
            sim_lc_meta['SM_seed'] = [sn.sct_mod_seed for sn in self.sn_list]

        if 'mw_dust' in self.config['model_config']:
            sim_lc_meta['MW_EBV'] = [sn.mw_ebv for sn in self.sn_list]

        write_file = self.config['data']['write_path'] + self.sim_name + '_fit.fits'
        ut.write_fit(sim_lc_meta, self.fit_res, write_file, sim_meta=self._get_primary_header())
