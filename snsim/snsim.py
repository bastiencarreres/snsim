"""Main module of the simulaiton package"""

import os
import pickle
import time
import yaml
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from numpy import power as pw
import snsim.utils as ut
from snsim.constants import SN_SIM_PRINT, VCMB, RA_CMB, DEC_CMB
import snsim.scatter as sct
import snsim.sim_class as scls

class Simulator:
    """Simulation class using a config file config.yml

    Parameters
    ----------
    param_dic : dict / str
        The configuration yaml file /
        The dictionnary containing all simulation parameters.

    Attributes
    ----------
    sim_cfg : dict
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
    __time_rate_bins()
        Give the time rate SN/years in redshift bins.
    __gen_n_sn(rand_seed)
        Generate the number of SN with Poisson law.
    simulate()
        Launch the simulation.
    __cadence_sim()
        Simulaton where the number of SN observed is determined by
        survey properties and poisson law.
    __fix_nsn_sim()
        Simulation where the number of SN is fixed.
    __get_primary_header()
        Generate the primary header of sim fits file.
    __write_sim()
        Write sim lightcurves in fits or/and pkl format(s).
    plot_lc(sn_ID, mag = False, zp = 25., plot_sim = True, plot_fit = False)
        Plot the given SN lightcurve.
    plot_ra_dec(self, plot_vpec=False, **kwarg):
        Plot a mollweide map of ra, dec.
    fit_lc(sn_ID = None)
        Fit all or just one SN lightcurve(s).
    write_fit()
        Write fits results in fits format.

    NOTES
    -----
    Remarks :

    - obs_file and db_file are optional but you must set one of the two!!!
    - If the name of bands in the obs/db file doesn't match sncosmo bands
    you can use the key band_dic to translate filters names
    - If you don't set the filter name item in nep_cut, the cut apply to all the bands
    - For wavelength dependent model, nomanclature follow arXiv:1209.2482
        Possibilities are :
        -> 'G10' for Guy et al. 2010 model,
        -> 'C11' or 'C11_0' for Chotard et al. model with correlation between
        U' and U = 0, 'C11_1' for Cor(U',U) = 1 and 'C11_2' for Cor(U',U) = -1

    yaml file format :

    +----------------------------------------------------------------------------------+
    | data :                                                                           |
    |     write_path: '/PATH/TO/OUTPUT'                                                |
    |     sim_name: 'NAME OF SIMULATION'                                               |
    |     write_format: 'format' or ['format1','format2']                              |
    | survey_config: #(Optional -> use obs_file)                                       |
    |     survey_file: '/PATH/TO/FILE'                                                 |
    |     band_dic: {'r':'ztfr','g':'ztfg','i':'ztfi'} #(Optional -> if bandname in    |
    |     add_data: ['keys1', 'keys2', ...] add db file keys to metadata               |
    |     survey_cut: {'key1': ['conditon1','conditon2',...], 'key2': ['conditon1']}   |
    |     duration: DURATION OF THE SURVEY (Optional, default given by survey file)    |
    |     zp: INSTRUMENTAL ZEROPOINT                                                   |
    |     ra_size: RA FIELD SIZE                                                       |
    |     dec_size: DEC FIELD SIZE                                                     |
    |     gain: CCD GAIN e-/ADU                                                        |
    | sn_gen:                                                                          |
    |     n_sn: NUMBER OF SN TO GENERATE (Otional)                                     |
    |     sn_rate: rate of SN/Mpc^3/year (Optional, default=3e-5)                      |
    |     rate_pw: rate = sn_rate*(1+z)^rate_pw (Optional, default=0)                  |
    |     nep_cut: [[nep_min1,Tmin,Tmax],[nep_min2,Tmin2,Tmax2,'filter1'],...] EP CUTS |
    |     randseed: RANDSEED TO REPRODUCE SIMULATION #(Optional)                       |
    |     z_range: [ZMIN,ZMAX]                                                         |
    |     M0: SN ABSOLUT MAGNITUDE                                                     |
    |     mag_smear: SN INTRINSIC SMEARING                                             |
    |     smear_mod: 'G10','C11_i' USE WAVELENGHT DEP MODEL FOR SN INT SCATTERING      |
    | cosmology:                                                                       |
    |     Om0: MATTER DENSITY                                                          |
    |     H0: HUBBLE CONSTANT                                                          |
    | cmb:                                                                             |
    |     v_cmb: OUR PECULIAR VELOCITY #(Optional, default = 369.82 km/s)              |
    |     ra_cmb: RIGHT ASCENSION OF CMB DIPOLE #(Optional, default = 266.81)          |
    |     dec_cmb: DECLINAISON OF CMB DIPOLE #(Optional, default = 48.253)             |
    | model_config:                                                                    |
    |     model_name:                                                                  |
    |     model_dir: '/PATH/TO/SALT/MODEL'                                             |
    |     alpha: STRETCH CORRECTION = alpha*x1                                         |
    |     beta: COLOR CORRECTION = -beta*c                                             |
    |     mean_x1: MEAN X1 VALUE                                                       |
    |     mean_c: MEAN C VALUE                                                         |
    |     sig_x1: SIGMA X1 or [SIGMA_X1_LOW, SIGMA_X1_HIGH]                            |
    |     sig_c: SIGMA C or [SIGMA_C_LOW, SIGMA_C_HIGH]                                |
    | vpec_gen:                                                                        |
    |     mean_vpec: MEAN SN PECULIAR VEL                                              |
    |     sig_vpec: SIGMA VPEC                                                         |
    | host_file: 'PATH/TO/HOSTFILE'  #(Optional)                                       |
    |                                                                                  |
    +----------------------------------------------------------------------------------+
    """

    def __init__(self, param_dic):
        # Load param dict from a yaml or by using launch_script.py
        if isinstance(param_dic, dict):
            self.sim_cfg = param_dic
            if 'yaml_path' in param_dic:
                self._yml_path = param_dic['yaml_path']
            else:
                self._yml_path = 'No config file'

        elif isinstance(param_dic, str):
            self._yml_path = param_dic
            with open(self._yml_path, "r") as f:
                self.sim_cfg = yaml.safe_load(f)

        # Check if there is a db_file
        if 'survey_config' not in self.sim_cfg:
            raise RuntimeError("Set a survey_file -> type help(sn_sim) to print the syntax")

        # cadence sim or n fixed
        if 'n_sn' in self.sim_cfg['sn_gen']:
            self._use_rate = False
        else:
            self._use_rate = True

        self._sn_list = None
        self._fit_res = None
        self._random_seed = None
        self._host = None
        self._cosmology = FlatLambdaCDM(**self.sim_cfg['cosmology'])
        self._obs = scls.SurveyObs(self.sim_cfg['survey_config'])
        self._generator = scls.SnGen(self.sn_int_par,
                                     self.sim_cfg['model_config'],
                                     self.cmb,
                                     self.cosmology,
                                     self.vpec_dist,
                                     host=self.host)

    @property
    def sim_name(self):
        """Get sim name"""
        return self.sim_cfg['data']['sim_name']

    @property
    def vpec_dist(self):
        """Get vpec option"""
        if 'vpec_gen' in self.sim_cfg:
            return self.sim_cfg['vpec_gen']
        elif 'host_file' in self.sim_cfg:
            return None
        else:
            return {'mean_vpec': 0., 'sig_vpec': 0.}

    @property
    def cmb(self):
        if 'cmb' in  self.sim_cfg:
            if 'vcmb' in self.sim_cfg['cmb']:
                vcmb = self.sim_cfg['cmb']['vcmb']
            else:
                vcmb = VCMB
            if 'ra_cmb' in self.sim_cfg['cmb']:
                ra_cmb = self.sim_cfg['cmb']['ra_cmb']
            else:
                ra_cmb =  RA_CMB
            if 'dec_cmb' in self.sim_cfg['cmb']:
                dec_cmb = self.sim_cfg['cmb']['dec_cmb']
            else:
                dec_cmb = DEC_CMB
        else:
            vcmb = VCMB
            ra_cmb = RA_CMB
            dec_cmb = DEC_CMB
        return {'v_cmb': vcmb, 'ra_cmb': ra_cmb, 'dec_cmb': dec_cmb}

    @property
    def n_sn(self):
        """Get number of sn simulated"""
        if self._sn_list is None:
            print('You have to run the simulation')
            return
        else:
            return len(self._sn_list)

    @property
    def model_name(self):
        """Get the name of sn model used"""
        return self.sim_cfg['model_config']['model_name']

    @property
    def sn_list(self):
        """Get the list of simulated sn"""
        return self._sn_list

    @property
    def fit_res(self):
        """Get the sn fit results"""
        return self._fit_res

    @property
    def sn_int_par(self):
        """Get the intrinsic parameters of SN Ia"""
        int_params = {'M0': self.sim_cfg['sn_gen']['M0'],
                      'mag_smear': self.sim_cfg['sn_gen']['mag_smear']}
        if 'smear_mod' in self.sim_cfg['sn_gen']:
            int_params['smear_mod'] = self.sim_cfg['sn_gen']['smear_mod']

        return int_params

    @property
    def cosmology(self):
        """Get astropy cosmological model used in simulation"""
        if not ut.is_same_cosmo_model(self.sim_cfg['cosmology'],self._cosmology):
            self._cosmology = FlatLambdaCDM(**self.sim_cfg['cosmology'])
        return self._cosmology

    @property
    def obs(self):
        """Get the ObsTable object of the simulation """
        if self._obs._survey_config != self.sim_cfg['survey_config']:
            self._obs = scls.SurveyObs(self.sim_cfg['survey_config'])
        return self._obs


    @property
    def generator(self):
        """Get the SNGen object of the simulation """
        not_same = (self._generator.sn_int_par != self.sn_int_par)
        not_same *= (self.sim_cfg['model_config'] != self._generator.model_config)
        not_same *= (not ut.is_same_cosmo_model(self.sim_cfg['cosmology'],self._cosmology))
        if not_same:
            self._generator = scls.SnGen(self.sn_int_par,
                                         self.sim_cfg['model_config'],
                                         self.cmb,
                                         cosmology=self.cosmology,
                                         host=self.host)
        return self._generator

    @property
    def host(self):
        """Get the SnHost object of the simulation """
        if self._host is not None:
            return self._host
        elif 'host_file' in self.sim_cfg:
            self._host = scls.SnHost(self.sim_cfg['host_file'], self.z_range)
            return self._host
        else:
            return None

    @property
    def nep_cut(self):
        """Get the list of epochs cuts"""
        snc_mintime, snc_maxtime = self.generator.snc_model_time
        if 'nep_cut' in self.sim_cfg['sn_gen']:
            nep_cut = self.sim_cfg['sn_gen']['nep_cut']
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
        """Get primary random seed of the simulation"""
        if 'randseed' in self.sim_cfg['sn_gen']:
            return int(self.sim_cfg['sn_gen']['randseed'])
        elif self._random_seed is None:
            self._random_seed = np.random.randint(low=1000, high=100000)
        return self._random_seed

    @property
    def z_range(self):
        """Get the simulation cosmological redshift range """
        return self.sim_cfg['sn_gen']['z_range']

    @property
    def survey_duration(self):
        """Get the survey duration"""
        if 'n_sn' in self.sim_cfg['sn_gen']:
            return None
        elif 'duration' not in self.sim_cfg['survey_config']:
            duration = (self.obs.mintime() - self.obs.maxtime()) / 365.25
        else:
            duration = self.sim_cfg['survey_config']['duration']
        return duration

    @property
    def z_span(self):
        """Get the redshift bins parameters"""
        z_min, z_max = self.z_range
        rate_pw = self.sim_cfg['sn_gen']['rate_pw']
        dz = (z_max - z_min) / (100 * (1 + rate_pw * z_max))
        if self.host is not None:
            host_max_dz = self.host.max_dz
            dz = np.max([dz, 2 * host_max_dz])
        n_bins = int((z_max - z_min) / dz)
        z_bins = np.linspace(z_min, z_max - dz, n_bins)
        return {'z_bins': z_bins, 'dz': dz, 'n_bins': n_bins}

    @property
    def sn_rate_z0(self):
        """Get the sn rate parameters"""
        if 'sn_rate' and 'rate_pw' in self.sim_cfg['sn_gen']:
            return float(self.sim_cfg['sn_gen']['sn_rate']), self.sim_cfg['sn_gen']['rate_pw']
        else:
            return 3e-5, 0

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

    def __time_rate_bins(self):
        """Give the time rate SN/years in redshift bins.

        Returns
        -------
        numpy.ndarray(float)
            Numpy array containing the time rate in each redshift bin.

        """
        z = self.z_span['z_bins']
        dz = self.z_span['dz']

        rate = self.sn_rate(z + 0.5 * dz)  # Rate in Nsn/Mpc^3/year
        shell_vol = 4 * np.pi / 3 * (pw(self.cosmology.comoving_distance(
            z + dz).value, 3) - pw(self.cosmology.comoving_distance(z).value, 3))
        time_rate = rate * shell_vol
        return time_rate

    def __gen_n_sn(self, rand_gen):
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
        time_rate = self.__time_rate_bins()
        return rand_gen.poisson(self.survey_duration * time_rate)

    def simulate(self):
        """Launch the simulation.

        Returns
        -------
        None

        Notes
        -----
        Simulation routine :
        1- Use either __cadence_sim() or __gen_n_sn()
        to run the simulation
        2- Gen all SN parameters inside SNGen class or/and SnHost class
        3- Check if SN pass cuts and then generate the lightcurves.
        4- Write LC to fits/pkl file(s)

        """

        print(SN_SIM_PRINT)
        print('-----------------------------------------------------------')
        print(f'SIM NAME : {self.sim_name}')
        print(f'CONFIG FILE : {self._yml_path}')
        print(f"OBS FILE : {self.sim_cfg['survey_config']['survey_file']}")
        if 'host_file' in self.sim_cfg :
            print(f"HOST FILE : {self.sim_cfg['host_file']}")
        print(f"SN SIM MODEL: {self.model_name} from {self.sim_cfg['model_config']['model_dir']}")
        print(f"SIM WRITE DIRECTORY : {self.sim_cfg['data']['write_path']}")
        print(f'SIMULATION RANDSEED : {self.rand_seed}')

        print(f'-----------------------------------------------------------\n')

        if self._use_rate:
            duration_str = f'Survey duration is {self.survey_duration} year(s)'
            print(
                f"Generate with a rate of r_v = {self.sn_rate_z0[0]}*(1+z)^{self.sn_rate_z0[1]} SN/Mpc^3/year")
            print(duration_str + '\n')
        else:
            print(f"Generate {self.sim_cfg['sn_gen']['n_sn']} SN Ia")

        print(f'-----------------------------------------------------------\n')

        if 'survey_cit' in self.sim_cfg['survey_config']:
            for k, v in self.obs_parameters['db_cut'].items():
                conditions_str = ''
                for cond in v:
                    conditions_str += str(cond) + ' OR '
                conditions_str = conditions_str[:-4]
                print(f'Select {k}: ' + conditions_str)
        else:
            print('No db cut')

        print(f'\n-----------------------------------------------------------\n')

        print("SN ligthcurve cuts :")

        for cut in self.nep_cut:
            print_cut = f'- At least {cut[0]} epochs between {cut[1]} and {cut[2]}'
            if len(cut) == 4:
                print_cut += f' in {cut[3]} band'
            print(print_cut)

        print(f'\n-----------------------------------------------------------\n')

        sim_time = time.time()
        self._sn_list = []
        #-- Create the random generator object with the rand seed
        rand_gen = np.random.default_rng(self.rand_seed)
        if self._use_rate:
            self.__cadence_sim(rand_gen)
        else:
            self.__fix_nsn_sim(rand_gen)
        l = f'{len(self._sn_list)} SN lcs generated in {time.time() - sim_time:.1f} seconds'
        print(l)

        print(f'\n-----------------------------------------------------------\n')

        write_time = time.time()
        self.__write_sim()
        l = f'Sim file write in {time.time() - write_time:.1f} seconds'
        print(l)

        print(f'\n-----------------------------------------------------------\n')

        print('OUTPUT FILE(S) : ')
        if isinstance(self.sim_cfg['data']['write_format'], str):
            print(self.sim_cfg['data']['write_path']
                  + self.sim_name
                  + '.'
                  + self.sim_cfg['data']['write_format'])
        else:
            for f in self.sim_cfg['data']['write_format']:
                print('- '
                      + self.sim_cfg['data']['write_path']
                      + self.sim_name
                      + '.'
                      + f)

    def __cadence_sim(self, rand_gen):
        """Simulaton where the number of SN observed is determined by
        survey properties and poisson law..

        Parameters
        ----------
        rand_gen : numpy.random.default_rng
            Numpy random generator.

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

        #n_sn_seed, sn_gen_seed = np.random.default_rng(
        #    self.rand_seed).integers(
        #    low=1000, high=100000, size=2)
        n_sn = self.__gen_n_sn(rand_gen)
        #sn_bins_seed = np.random.default_rng(sn_gen_seed).integers(
        #    low=1000, high=100000, size=np.sum(n_sn))

        SN_ID = 0
        for n, z in zip(n_sn, self.z_span['z_bins']):
            sn_list_tmp = self.generator(n, [z, z + self.z_span['dz']],[self.obs.mintime, self.obs.maxtime], rand_gen)
            for sn in sn_list_tmp:
                sn.epochs = self.obs.epochs_selection(sn)
                if sn.pass_cut(self.nep_cut):
                    sn.gen_flux(rand_gen)
                    sn.ID = SN_ID
                    SN_ID += 1
                    self._sn_list.append(sn)

    def __fix_nsn_sim(self, rand_gen):
        """Simulation where the number of SN is fixed.

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
        while len(self._sn_list) < self.sim_cfg['sn_gen']['n_sn']:
            sn = self.generator(1, self.z_range, [self.obs.mintime, self.obs.maxtime], rand_gen)[0]
            sn.epochs = self.obs.epochs_selection(sn)
            if sn.pass_cut(self.nep_cut):
                sn.gen_flux(rand_gen)
                sn.ID = SN_ID
                SN_ID += 1
                self._sn_list.append(sn)
            elif raise_trigger > 2 * len(self.obs.obs_table['expMJD']):
                raise RuntimeError('Cuts are too stricts')
            else:
                raise_trigger += 1

    def __get_primary_header(self):
        """Generate the primary header of sim fits file..

        Returns
        -------
        None

        """
        header = {'n_sn': self.n_sn,
                  'M0': self.sim_cfg['sn_gen']['M0'],
                  'sigM': self.sim_cfg['sn_gen']['mag_smear'],
                  **self.sim_cfg['cosmology']}

        if self.host is None:
            header['m_vp'] = self.sim_cfg['vpec_gen']['mean_vpec']
            header['s_vp'] = self.sim_cfg['vpec_gen']['sig_vpec']

        if self.model_name == 'salt2' or self.model_name == 'salt3':
            fits_dic = {'model_name': 'Mname',
                        'alpha': 'alpha',
                        'beta': 'beta',
                        'mean_x1': 'm_x1',
                        'sig_x1': 's_x1',
                        'mean_c': 'm_c',
                        'sig_c': 's_c'
                        }

        for k, v in fits_dic.items():
            header[v] = self.sim_cfg['model_config'][k]

        if 'smear_mod' in self.sim_cfg['sn_gen']:
            header['Smod'] = self.sim_cfg['sn_gen']['smear_mod']
        return header

    def __write_sim(self):
        """Write sim lightcurves in fits or/and pkl format(s).

        Returns
        -------
        None
            Just write sim into a file

        """

        write_path = self.sim_cfg['data']['write_path']
        sim_header = self.__get_primary_header()
        if 'fits' in self.sim_cfg['data']['write_format']:
            lc_hdu_list = (sn.get_lc_hdu() for sn in self._sn_list)
            hdu_list = fits.HDUList(
                [fits.PrimaryHDU(header=fits.Header(sim_header))] + list(lc_hdu_list))

            hdu_list.writeto(
                write_path +
                self.sim_name +
                '.fits',
                overwrite=True)

        # Export lcs as pickle
        if 'pkl' in self.sim_cfg['data']['write_format']:
            sim_lc = [sn.sim_lc for sn in self._sn_list]
            sn_pkl = scls.SnSimPkl(sim_lc, sim_header)
            with open(write_path + self.sim_name + '_lcs.pkl', 'wb') as file:
                pickle.dump(sn_pkl, file)

    def plot_lc(self, sn_ID, mag=False, zp=25., plot_sim=True, plot_fit=False):
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
            if self.fit_res[sn_ID] is None:
                print('This SN was not fitted, launch fit')
                self.fit_lc(sn_ID)
            if self.fit_res[sn_ID] is np.nan:
                print('This sn has no fit results')
                return
            f_model = ut.init_sn_model(self.model_name, self.sim_cfg['model_config']['model_dir'])
            x0, x1, c = self.fit_res[sn_ID]['parameters'][2:]
            f_model.set(t0=sn.sim_t0, z=sn.z, x0=x0, x1=x1, c=c)
            cov_x0_x1_c = self.fit_res[sn_ID]['covariance'][1:, 1:]
            residuals = True
        else:
            f_model = None
            cov_x0_x1_c = None
            residuals = False

        ut.plot_lc(sn.sim_lc, mag=mag,
                   snc_sim_model=s_model,
                   snc_fit_model=f_model,
                   fit_cov=cov_x0_x1_c,
                   zp=zp,
                   residuals=residuals)

    def get_sn_par(self, key):
        """Get an array of a sim_lc data"""
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
            field_dic = self.obs._field_dic
            field_size = self.obs.field_size
            field_list = np.unique(field_list)
        else:
            field_dic = None
            field_size = None
            field_list = None
        ut.plot_ra_dec(np.asarray(ra), np.asarray(dec), vpec, field_list, field_dic, field_size, **kwarg)

    def fit_lc(self, sn_ID=None):
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

        fit_model = ut.init_sn_model(self.model_name, self.sim_cfg['model_config']['model_dir'])

        if self.model_name in ('salt2', 'salt3'):
            fit_par = ['t0', 'x0', 'x1', 'c']

        if sn_ID is None:
            for i, sn in enumerate(self.sn_list):
                if self._fit_res[i] is None:
                    fit_model.set(z=sn.z)
                    self._fit_res[i] = ut.snc_fitter(sn.sim_lc, fit_model, fit_par)
        else:
            fit_model.set(z=self.sn_list[sn_ID].z)
            self._fit_res[sn_ID] = ut.snc_fitter(self.sn_list[sn_ID].sim_lc, fit_model, fit_par)

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
                       'sim_t0': [sn.sim_t0 for sn in self.sn_list] }

        if self.model_name == 'salt2' or self.model_name == 'salt3':
            sim_lc_meta['sim_mb'] = [sn.sim_mb for sn in self.sn_list]
            sim_lc_meta['sim_x1'] = [sn.sim_x1 for sn in self.sn_list]
            sim_lc_meta['sim_c'] = [sn.sim_c for sn in self.sn_list]
            sim_lc_meta['m_smear'] = [sn.mag_smear for sn in self.sn_list]

        sim_meta = {
            'n_sn': len(
                self.sn_list),
            'Mname': self.model_name,
            'M0': self.sim_par['sn_model_par']['M0'],
            **self.sim_par['cosmo']}

        if self.model_name in ('salt2', 'salt3'):
            sim_meta['alpha'] = self.sim_cfg['model_config']['alpha']
            sim_meta['beta'] = self.sim_cfg['model_config']['beta']
            sim_meta['m_x1'] = self.sim_cfg['model_config']['mean_x1']
            sim_meta['s_x1'] = self.sim_cfg['model_config']['sig_x1']
            sim_meta['m_c'] = self.sim_cfg['model_config']['mean_c']
            sim_meta['s_c'] = self.sim_cfg['model_config']['sig_c']

            sim_lc_meta['sim_x0'] = [sn.sim_x0 for sn in self.sn_list]
            sim_lc_meta['sim_mb'] = [sn.sim_mb for sn in self.sn_list]
            sim_lc_meta['sim_x1'] = [sn.sim_x1 for sn in self.sn_list]
            sim_lc_meta['sim_c'] = [sn.sim_c for sn in self.sn_list]

        if 'smear_mod' in self.sim_cfg['sn_gen']:
            sim_meta['SMod'] = self.sim_cfg['sn_gen']['smear_mod']
            sim_lc_meta['SM_seed'] = [sn.smear_mod_seed for sn in self.sn_list]

        write_file = self.sim_cfg['data']['write_path'] + self.sim_name + '_fit.fits'
        ut.write_fit(sim_lc_meta, self.fit_res, write_file, sim_meta=sim_meta)


class OpenSim:
    """This class allow to open simulation file, make plot and run the fit.

    Parameters
    ----------
    sim_file : str
        Path to the simulation file fits/pkl.
    model_dir : str
        Path to the .model used during simulation

    Attributes
    ----------
    _file_path : str
        The path of the simulation file.
    _file_ext : str
        sim file extension.
    _sim_lc : list(astropy.Table)
        List containing the simulated lightcurves.
    _header : dict
        A dict containing simulation meta.
    _model_dir : str
        A copy of input model dir.
    _fit_model : sncosmo.Model
        The model used to fit the lightcurves.
    _fit_res : list(sncomso.utils.Result)
        The reuslts of sncosmo fit.

    Methods
    -------
    __init_sim_lc()
        Extract data from file.
    plot_lc(sn_ID, mag = False, zp = 25., plot_sim = True, plot_fit = False)
        Plot the given SN lightcurve.
    plot_ra_dec(self, plot_vpec=False, **kwarg):
        Plot a mollweide map of ra, dec.
    fit_lc(sn_ID = None)
        Fit all or just one SN lightcurve(s).
    write_fit()
        Write fits results in fits format.


    """

    def __init__(self, sim_file, model_dir):
        '''Copy some function of snsim to allow to use sim file'''
        self._file_path, self._file_ext = os.path.splitext(sim_file)
        self._sim_lc, self._header = self.__init_sim_lc()
        self._model_dir = model_dir
        self._fit_model = ut.init_sn_model(self.header['Mname'], model_dir)
        self._fit_res = None

    def __init_sim_lc(self):
        if self._file_ext == '.fits':
            sim_lc = []
            with fits.open(self._file_path + self._file_ext) as sf:
                header = sf[0].header
                for hdu in sf[1:]:
                    data = hdu.data
                    tab = Table(data)
                    tab.meta = hdu.header
                    sim_lc.append(tab)

        elif self._file_ext == '.pkl':
            with open(self._file_path + self._file_ext, 'rb') as f:
                sn_pkl = pickle.load(f)
                sim_lc = sn_pkl.sim_lc
                header = sn_pkl.header
        return sim_lc, header

    @property
    def sim_lc(self):
        """Get sim_lc list """
        return self._sim_lc

    @property
    def header(self):
        """Get header dict """
        return self._header

    @property
    def fit_res(self):
        """Get fit results list"""
        return self._fit_res

    def fit_lc(self, sn_ID=None):
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
        if self._fit_res is None:
            self._fit_res = [None] * len(self.sim_lc)
        fit_model = self._fit_model.__copy__()
        model_name = self.header['Mname']
        if model_name in ('salt2', 'salt3'):
            fit_par = ['t0', 'x0', 'x1', 'c']

        if sn_ID is None:
            for i, lc in enumerate(self.sim_lc):
                if self._fit_res[i] is None:
                    fit_model.set(z=lc.meta['z'])
                    self._fit_res[i] = ut.snc_fitter(lc, fit_model, fit_par)
        else:
            fit_model.set(z=self.sim_lc[sn_ID].meta['z'])
            self._fit_res[sn_ID] = ut.snc_fitter(self.sim_lc[sn_ID], fit_model, fit_par)

    def plot_lc(self, sn_ID, mag=False, zp=25., plot_sim=True, plot_fit=False):
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

        Returns
        -------
        None
            Just plot the SN lightcurve !

        Notes
        -----
        Use plot_lc from utils.

        """
        lc = self.sim_lc[sn_ID]

        if plot_sim:
            model_name = self.header['Mname']

            s_model = ut.init_sn_model(model_name, self._model_dir)
            dic_par = {'z': lc.meta['z'],
                       't0': lc.meta['sim_t0']}

            if model_name in ('salt2', 'salt3'):
                dic_par['x0'] = lc.meta['sim_x0']
                dic_par['x1'] = lc.meta['sim_x1']
                dic_par['c'] = lc.meta['sim_c']

            s_model.set(**dic_par)

            if 'Smod' in self.header:
                s_model = sct.init_sn_smear_model(s_model, self.header['Smod'])
                par_rd_name = self.header['Smod'][:3] + '_RndS'
                s_model.set(**{par_rd_name: lc.meta[par_rd_name]})
        else:
            s_model = None

        if plot_fit:
            if self.fit_res is None:
                print('This SN was not fitted, launch fit')
                self.fit_lc(sn_ID)
            elif self.fit_res[sn_ID] is None:
                print('This SN was not fitted, launch fit')
                self.fit_lc(sn_ID)
            if self.fit_res[sn_ID] is np.nan:
                print('This sn has no fit results')
                return
            f_model = ut.init_sn_model(self.header['Mname'], self._model_dir)
            x0, x1, c = self.fit_res[sn_ID]['parameters'][2:]
            f_model.set(t0=self.sim_lc[sn_ID].meta['sim_t0'],
                        z=self.sim_lc[sn_ID].meta['z'], x0=x0, x1=x1, c=c)
            cov_x0_x1_c = self.fit_res[sn_ID]['covariance'][1:, 1:]
            residuals = True
        else:
            f_model = None
            cov_x0_x1_c = None
            residuals = False

        ut.plot_lc(self.sim_lc[sn_ID],
                   mag=mag,
                   snc_sim_model=s_model,
                   snc_fit_model=f_model,
                   fit_cov=cov_x0_x1_c,
                   zp=zp,
                   residuals=residuals)

    def plot_ra_dec(self, plot_vpec=False, **kwarg):
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
        for lc in self.sim_lc:
            ra.append(lc.meta['ra'])
            dec.append(lc.meta['dec'])
            if plot_vpec:
                vpec.append(lc.meta['vpec'])

        ut.plot_ra_dec(np.asarray(ra), np.asarray(dec), vpec, **kwarg)

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
        if self.fit_res is None:
            print('Perform fit before write')
            self.fit_lc()
        for i, res in enumerate(self.fit_res):
            if res is None:
                self.fit_lc(self.sim_lc[i].meta['sn_id'])

        sim_lc_meta = {'sn_id': [lc.meta['sn_id'] for lc in self.sim_lc],
                       'ra': [lc.meta['ra'] for lc in self.sim_lc],
                       'dec': [lc.meta['dec'] for lc in self.sim_lc],
                       'vpec': [lc.meta['vpec'] for lc in self.sim_lc],
                       'zpec': [lc.meta['zpec'] for lc in self.sim_lc],
                       'z2cmb': [lc.meta['z2cmb'] for lc in self.sim_lc],
                       'zcos': [lc.meta['zcos'] for lc in self.sim_lc],
                       'zCMB': [lc.meta['zCMB'] for lc in self.sim_lc],
                       'zobs': [lc.meta['z'] for lc in self.sim_lc],
                       'sim_mu': [lc.meta['sim_mu'] for lc in self.sim_lc]}

        model_name = self.header['Mname']
        if model_name in ('salt2', 'salt3'):
            sim_lc_meta['sim_mb'] = [lc.meta['sim_mb'] for lc in self.sim_lc]
            sim_lc_meta['sim_x1'] = [lc.meta['sim_x1'] for lc in self.sim_lc]
            sim_lc_meta['sim_c'] = [lc.meta['sim_c'] for lc in self.sim_lc]
            sim_lc_meta['m_smear'] = [lc.meta['m_smear'] for lc in self.sim_lc]

        if 'Smod' in self.header:
            sim_lc_meta['SM_seed'] = [lc.meta[self.header['Smod'][:3] + '_RndS']
                                      for lc in self.sim_lc]

        write_file = self._file_path + '_fit.fits'
        ut.write_fit(sim_lc_meta, self.fit_res, write_file, sim_meta=self.header)
