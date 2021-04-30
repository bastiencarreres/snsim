import sncosmo as snc
import numpy as np
import astropy.units as u
from numpy import power as pw
from astropy.table import Table
from astropy.io import fits
import yaml
import pickle
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import time
import sqlite3
from . import sim_utils as su
from . import scatter as sct


class sn_sim:
    def __init__(self, param_dic):
        '''Initialisation of the simulation class with the config file
        config.yml

        NOTE : - obs_file and db_file are optional but you must set one of the two!!!
               - If the name of bands in the obs/db file doesn't match sncosmo bands
            you can use the key band_dic to translate filters names
               - If you don't set the filter name item in nep_cut, the cut apply to all the bands
               - For wavelength dependent model, nomanclature follow arXiv:1209.2482 -> Possibility are
            'G10' for Guy et al. 2010 model, 'C11' or 'C11_0' for Chotard et al. model with correlation
            between U' and U = 0, 'C11_1' for Cor(U',U) = 1 and 'C11_2' for Cor(U',U) = -1

        +----------------------------------------------------------------------------------+
        | data :                                                                           |
        |     write_path: '/PATH/TO/OUTPUT'                                                |
        |     sim_name: 'NAME OF SIMULATION'                                               |
        |     band_dic: {'r':'ztfr','g':'ztfg','i':'ztfi'} #(Optional -> if bandname in    |
        | db/obs file doesn't correpond to those in sncosmo registery)                     |
        |     obs_config_path: '/PATH/TO/OBS/FILE' #(Optional -> use db_file)              |
        | db_config: #(Optional -> use obs_file)                                           |
        |     dbfile_path: '/PATH/TO/FILE'                                                 |
        |     db_cut: {'key1': ['conditon1','conditon2',...], 'key2': ['conditon1'],...}   |
        |     zp: INSTRUMENTAL ZEROPOINT                                                   |
        |     ra_size: RA FIELD SIZE                                                       |
        |     dec_size: DEC FIELD SIZE                                                     |
        |     gain: CCD GAIN e-/ADU                                                        |
        | sn_gen:                                                                          |
        |     n_sn: NUMBER OF SN TO GENERATE (Otional)                                     |
        |     sn_rate: rate of SN/Mpc^3/year (Optional, default=3e-5)                      |
        |     rate_pw: rate = sn_rate*(1+z)^rate_pw (Optional, default=0)                  |
        |     duration: DURATION OF THE SURVEY (Optional, default given by cadence file)   |
        |     nep_cut: [[nep_min1,Tmin,Tmax],[nep_min2,Tmin2,Tmax2,'filter1'],...] EP CUTS |
        |     randseed: RANDSEED TO REPRODUCE SIMULATION #(Optional)                       |
        |     z_range: [ZMIN,ZMAX]                                                         |
        |     v_cmb: OUR PECULIAR VELOCITY #(Optional, default = 369.82 km/s)              |
        |     M0: SN ABSOLUT MAGNITUDE                                                     |
        |     mag_smear: SN INTRINSIC SMEARING                                             |
        |     smear_mod: 'G10','C11_i' USE WAVELENGHT DEP MODEL FOR SN INT SCATTERING      |
        | cosmology:                                                                       |
        |     Om: MATTER DENSITY                                                           |
        |     H0: HUBBLE CONSTANT                                                          |
        | salt_gen:                                                                        |
        |     version: 2 or 3                                                              |
        |     salt_dir: '/PATH/TO/SALT/MODEL'                                              |
        |     alpha: STRETCH CORRECTION = alpha*x1                                         |
        |     beta: COLOR CORRECTION = -beta*c                                             |
        |     mean_x1: MEAN X1 VALUE                                                       |
        |     mean_c: MEAN C VALUE                                                         |
        |     sig_x1: SIGMA X1                                                             |
        |     sig_c: SIGMA C                                                               |
        | vpec_gen:                                                                        |
        |     host_file: 'PATH/TO/HOSTFILE'                                                |
        |     mean_vpec: MEAN SN PECULIAR VEL                                              |
        |     sig_vpec: SIGMA VPEC                                                         |
        |                                                                                  |
        +----------------------------------------------------------------------------------+
        '''

    # Load param dict from a yaml or by using main.py
        if isinstance(param_dic, dict):
            self.sim_cfg = param_dic
            self.yml_path = param_dic['yaml_path']

        elif isinstance(param_dic, str):
            self.yml_path = param_dic
            with open(self.yml_path, "r") as f:
                self.sim_cfg = yaml.safe_load(f)

    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #----------------- DEFAULT VALUES -----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        # CMB values
        self.dec_cmb = 48.253
        self.ra_cmb = 266.81
        self.v_cmb = 369.82

    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #----------- data and db_config section -----------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        # Simulation parameters
        self.data_cfg = self.sim_cfg['data']

        # Condition to use obs_file or db_file
        if 'db_config' in self.sim_cfg and 'obs_config_path' in self.data_cfg:
            raise RuntimeError(
                "The simulation can't run with obs file and db file, just set one of the two")
        elif 'obs_config_path' in self.data_cfg:
            self.use_obs = True
        else:
            if 'db_config' in self.sim_cfg:
                self.use_obs = False
            else:
                raise RuntimeError(
                    "Set a db_file or a obs_file -> type help(sn_sim) to print the syntax")

        # Initialisation of db/obs_path
        if self.use_obs:
            self.obs_cfg_path = self.data_cfg['obs_config_path']
            self.open_obs_header()
        else:
            self.db_cfg = self.sim_cfg['db_config']
            self.db_file = self.db_cfg['dbfile_path']
            self.zp = self.db_cfg['zp']
            self.gain = self.db_cfg['gain']
            self.ra_size = self.db_cfg['ra_size']
            self.dec_size = self.db_cfg['dec_size']
            self.use_dbcut = False
            if 'db_cut' in self.db_cfg:
                self.use_dbcut = True
                self.db_cut = self.db_cfg['db_cut']


        self.write_path = self.data_cfg['write_path']
        self.sim_name = self.data_cfg['sim_name']

        # Band dic : band_name_obs/db_file -> band_name_sncosmo
        if 'band_dic' in self.data_cfg:
            self.band_dic = self.data_cfg['band_dic']
        else:
            self.band_dic = None

    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #----------------- sn_gen section -----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        self.sn_gen = self.sim_cfg['sn_gen']

        if 'n_sn' in self.sn_gen:
            self.n_sn = int(self.sn_gen['n_sn'])
            self.use_rate = False
            self.duration = None
        else:
            self.use_rate = True
            if 'sn_rate' in self.sn_gen:
                self.sn_rate = float(self.sn_gen['sn_rate'])
            else:
                self.sn_rate = 3e-5  # SN/Mpc^3/year
            if 'rate_pw' in self.sn_gen:
                self.rate_pw = self.sn_gen['rate_pw']
            else:
                self.rate_pw = 0  # No dependance in redshift
            if 'duration' in self.sn_gen:
                self.duration = self.sn_gen['duration']
            else:

                self.duration = None


    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #--------------- cosmomogy section ----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        # Cosmology parameters
        self.cosmo_cfg = self.sim_cfg['cosmology']
        self.cosmo = FlatLambdaCDM(
            H0=self.cosmo_cfg['H0'],
            Om0=self.cosmo_cfg['Om'])

    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #--------------- salt_gen section ----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        # Salt parameters
        self.salt_gen = self.sim_cfg['salt_gen']
        self.alpha = self.salt_gen['alpha']
        self.beta = self.salt_gen['beta']
        self.salt_dir = self.salt_gen['salt_dir']

        if self.salt_gen['version'] == 2:
            source = snc.SALT2Source(modeldir=self.salt_dir)
        elif self.salt_gen['version'] == 3:
            source = snc.SALT3Source(modeldir=self.salt_dir)
        else :
            raise RuntimeError("Support SALT version = 2 or 3")

        self.sim_model = snc.Model(source=source)
        self.fit_model = snc.Model(source=source)

        if 'smear_mod' in self.sn_gen:
            self.use_smear_mod = True
            if self.sn_gen['smear_mod'] == 'G10':
                self.smear_par_prefix='G10_'
                self.sim_model.add_effect(sct.G10(self.sim_model),'G10_','rest')
            elif self.sn_gen['smear_mod'][:3] == 'C11':
                self.smear_par_prefix='C11_'
                if self.sn_gen['smear_mod'] == ('C11' or 'C11_0'):
                    self.sim_model.add_effect(sct.C11(self.sim_model),'C11_','rest')
                elif self.sn_gen['smear_mod'] == 'C11_1':
                    self.sim_model.add_effect(sct.C11(self.sim_model),'C11_','rest')
                    self.sim_model.set(C11_Cuu=1.)
                elif self.sn_gen['smear_mod'] == 'C11_2':
                    self.sim_model.add_effect(sct.C11(self.sim_model),'C11_','rest')
                    self.sim_model.set(C11_Cuu=-1.)

        else:
            self.use_smear_mod = False

    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #--------------- vpec_gen section -----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        self.vpec_gen = self.sim_cfg['vpec_gen']
        if 'host_file' in self.vpec_gen:
            self.use_host = True
            self.host_file = self.vpec_gen['host_file']
        else:
            self.use_host = False

    # Minimal nbr of epochs in LC
        if 'nep_cut' in self.sn_gen:
            if isinstance(self.sn_gen['nep_cut'], (int, float)):
                self.nep_cut = [
                    (self.sn_gen['nep_cut'],
                     self.sim_model.mintime(),
                     self.sim_model.maxtime())]
            elif isinstance(self.sn_gen['nep_cut'], (list)):
                self.nep_cut = self.sn_gen['nep_cut']

        else:
            self.nep_cut = [(1, self.sim_model.mintime(), self.sim_model.maxtime())]

        if 'v_cmb' in self.sn_gen:
            self.v_cmb = self.sn_gen['v_cmb']

    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #------------ Randseed Initialisation -------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        if 'randseed' in self.sn_gen:
            self.randseed = int(self.sn_gen['randseed'])
        else:
            self.randseed = np.random.randint(low=1000, high=100000)

        randseeds = np.random.default_rng(
            self.randseed).integers(
            low=1000, high=100000, size=11)
        self.randseeds = {'z_seed': randseeds[0],
                          't0_seed': randseeds[1],
                          'x0_seed': randseeds[2],
                          'x1_seed': randseeds[3],
                          'c_seed': randseeds[4],
                          'coord_seed': randseeds[5],
                          'vpec_seed': randseeds[6],
                          'smearM_seed': randseeds[7],
                          'sigflux_seed': randseeds[8],
                          'nsn_seed': randseeds[9],
                          'smearmod_seed': randseeds[10]
                          }
        return

    def simulate(self):
        '''Simulation routine :
        1- READ OBS/DB FILE
        2- GEN REDSHIFT AND SALT2 PARAM
        3- GEN LC FLUX WITH sncosmo
        4- WRITE LC TO A FITS FILE
        '''

        print(su.sn_sim_print)
        print('-----------------------------------')
        print(f'SIM NAME : {self.sim_name}')
        print(f'CONFIG FILE : {self.yml_path}')

        if self.use_obs:
            print(f'OBS FILE : {self.obs_cfg_path}')
        else:
            print(f'CADENCE FILE : {self.db_file}')

        print(f'SIM WRITE DIRECTORY : {self.write_path}')
        print(f'-----------------------------------\n')

        if not self.use_obs:
            if self.use_rate:
                if self.duration is None:
                    duration_str = f'Survey duration is given by cadence file'
                else:
                    duration_str = f'Survey duration is {self.duration} year(s)'

                print(f"Generate with a rate of r_v = {self.sn_rate}*(1+z)^{self.rate_pw} SN/Mpc^3/year")
                print(duration_str + '\n')
            else:
                print(f"Generate {self.n_sn} SN Ia")

            if self.use_dbcut:
                for k in self.db_cut:
                    conditions_str=''
                    for cond in self.db_cut[k]:
                        conditions_str+=str(cond)+' OR '
                    conditions_str=conditions_str[:-4]
                    print(f'Select {k}: '+conditions_str)
            else:
                print('No db cut')
            print('\n')

        if self.use_host:
            self.host = []
            with fits.open(self.host_file) as hduf:
                for hdu in hduf:
                    self.host.append(hdu.data)


        start_time = time.time()
        self.obs = []
        if self.use_obs:
            self.obs_header = []
            with fits.open(self.obs_cfg_path) as hduf:
                for hdu in hduf[1:]:
                    if self.band_dic is not None:
                        for i, b in hdu['band']:
                            hdu.data['band'][i] = self.band_dic[b]
                    self.obs.append(hdu.data)
                    self.obs_header.append(hdu.header)
        else:
            self.extract_from_db()

        sep2 = su.box_output(su.sep, '------------')
        line = f'OBS FILE read in {time.time()-start_time:.1f} seconds'
        print(su.sep)
        print(su.box_output(su.sep, line))
        print(sep2)
        sim_time = time.time()
        # Generate z, x0, x1, c
        self.gen_param_array()
        # Simulate for each obs
        self.gen_flux()

        l = f'{self.n_sn} SN lcs generated in {time.time() - sim_time:.1f} seconds'
        print(su.box_output(su.sep, l))
        print(sep2)

        write_time = time.time()
        self.write_sim()
        l = f'Sim file write in {time.time() - write_time:.1f} seconds'
        print(su.box_output(su.sep, l))
        print(sep2)
        l = f'SIMULATION TERMINATED in {time.time() - start_time:.1f} seconds'
        print(su.box_output(su.sep, l))
        print(su.sep)

        # Init fit_res_table
        self.fit_res = np.asarray(['No_fit'] * self.n_sn, dtype='object')
        return

    def gen_param_array(self):
        '''GENERATE Z,T0,SALT2 PARAMS'''

        # Init z range
        self.z_range = self.sn_gen['z_range']
        self.sigmaM = self.sn_gen['mag_smear']  # To change

        # Init vpec_gen
        self.mean_vpec = self.vpec_gen['mean_vpec']
        self.sig_vpec = self.vpec_gen['sig_vpec']

        # Init M0
        self.M0 = self.sn_gen['M0']

        # Init x1 and c
        self.mean_x1 = self.salt_gen['mean_x1']
        self.sig_x1 = self.salt_gen['sig_x1']

        self.mean_c = self.salt_gen['mean_c']
        self.sig_c = self.salt_gen['sig_c']

        # Redshift generation
        if not self.use_host and not self.use_rate:
            self.zcos = self.gen_redshift_cos()

        if self.use_obs:
            self.extract_coord()
        elif self.use_rate:
            self.cadence_sim()
        else:
            self.fix_nsn_sim()

        self.gen_z2cmb()
        self.gen_z_pec()
        self.zCMB = (1 + self.zcos) * (1 + self.zpec) - 1.
        self.zobs = (1 + self.zcos) * (1 + self.zpec) * (1 + self.z2cmb) - 1.

        # SALT2 params generation
        self.gen_sn_par()
        self.gen_sn_mag()

        self.params = [{'z': z,
                        't0': peak,
                        'x0': x0,
                        'x1': x1,
                        'c': c,
                        } for z,
                       peak,
                       x0,
                       x1,
                       c in zip(self.zobs,
                                self.sim_t0,
                                self.sim_x0,
                                self.sim_x1,
                                self.sim_c)]
        if self.use_smear_mod:
            self.smear_mod_seeds = np.random.default_rng(self.randseeds['smearmod_seed']).integers(low=1000, high=10000,size=self.n_sn)
            for par,s in zip(self.params,self.smear_mod_seeds):
                par[self.smear_par_prefix+'RndS'] = s
        return

    def open_obs_header(self):
        ''' Open the fits obs file header'''
        with fits.open(self.obs_cfg_path, 'readonly') as obs_fits:
            self.obs_header_main = obs_fits[0].header
            self.bands = self.obs_header_main['bands'].split()
        return

    def extract_coord(self):
        '''Extract ra and dec from obs file'''
        # extract ra dec from obs config
        self.ra = []
        self.dec = []
        for i in range(self.n_sn):
            obs = self.obs_header[i]
            self.ra.append(obs['RA'])
            self.dec.append(obs['DEC'])
        return

    def gen_redshift_cos(self, low=None, high=None, size=None, randseed=None):
        '''Function to get zcos, to be updated'''
        if high is None:
            high = self.z_range[1]
        if low is None:
            low = self.z_range[0]
        if size is None:
            size = self.n_sn
        if randseed is None:
            randseed = self.randseeds['z_seed']

        z = np.random.default_rng(randseed).uniform(
            low=low, high=high, size=size)
        return z

    def gen_coord(self, randseeds, size=1):
        '''Generate ra,dec uniform on the sphere'''
        ra_seed = randseeds[0]
        dec_seed = randseeds[1]
        ra = np.random.default_rng(ra_seed).uniform(
            low=0, high=2 * np.pi, size=size)
        dec_uni = np.random.default_rng(dec_seed).random(size=size)
        dec = np.arcsin(2 * dec_uni - 1)
        if size == 1:
            return ra[0], dec[0]
        else:
            return ra, dec

    def gen_z2cmb(self):
        # use ra dec to simulate the effect of our motion
        coordfk5 = SkyCoord(
            self.ra * u.deg,
            self.dec * u.deg,
            frame='fk5')  # coord in fk5 frame
        galac_coord = coordfk5.transform_to('galactic')
        self.ra_gal = galac_coord.l.rad - 2 * np.pi * \
            np.sign(galac_coord.l.rad) * (abs(galac_coord.l.rad) > np.pi)
        self.dec_gal = galac_coord.b.rad

        ss = np.sin(self.dec_gal) * np.sin(self.dec_cmb * np.pi / 180)
        ccc = np.cos(self.dec_gal) * np.cos(self.dec_cmb * np.pi / \
                     180) * np.cos(self.ra_gal - self.ra_cmb * np.pi / 180)
        self.z2cmb = (1 - self.v_cmb * (ss + ccc) / su.c_light_kms) - 1.
        return

    def gen_z_pec(self):
        if self.use_host:
            if 'vp_sight' in self.host.names:
                self.vpec = self.host['vp_sight']
        else:
            self.vpec = np.random.default_rng(
                self.randseeds['vpec_seed']).normal(
                loc=self.mean_vpec,
                scale=self.sig_vpec,
                size=self.n_sn)

        self.zpec = self.vpec / su.c_light_kms
        return

    def gen_sn_par(self):
        ''' Generate x1 and c for the SALT2 model'''
        self.sim_x1 = np.random.default_rng(
            self.randseeds['x1_seed']).normal(
            loc=self.mean_x1,
            scale=self.sig_x1,
            size=self.n_sn)
        self.sim_c = np.random.default_rng(
            self.randseeds['c_seed']).normal(
            loc=self.mean_c, scale=self.sig_c, size=self.n_sn)
        return

    def gen_sn_mag(self):
        ''' Generate x0/mB parameters for SALT2 '''
        self.mag_smear = np.random.default_rng(
            self.randseeds['smearM_seed']).normal(
            loc=0, scale=self.sigmaM, size=self.n_sn)
        self.sim_mu = 5 * np.log10((1 + self.zcos) * (1 + self.z2cmb) * pw(
            (1 + self.zpec), 2) * self.cosmo.comoving_distance(self.zcos).value) + 25
        # Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 +
        # beta*c : scattering due to color and stretch} + {intrinsic smearing}
        self.sim_mB = self.sim_mu + self.M0 - self.alpha * \
            self.sim_x1 + self.beta * self.sim_c + self.mag_smear
        self.sim_x0 = su.mB_to_x0(self.sim_mB)
        return

    def gen_flux(self):
        ''' Generate simulated flux '''
        lc_seeds = np.random.default_rng(
            self.randseeds['sigflux_seed']).integers(
            low=1000, high=100000, size=self.n_sn)
        self.sim_lc = []
        for obs, params, s in zip(self.obs, self.params, lc_seeds):
            lc = snc.realize_lcs(obs, self.sim_model, [params], scatter=False)[0]
            lc['flux'] = np.random.default_rng(s).normal(
                loc=lc['flux'], scale=lc['fluxerr'])
            if not self.use_obs:
                lc['subprogram'] = obs['subprogram']
            self.sim_lc.append(lc)
        return

    def extract_from_db(self):
        '''Read db file and extract relevant information'''
        self.obs_dic={}
        dbf = sqlite3.connect(self.db_file)

        keys = ['expMJD',
                'filter',
                'fieldRA',
                'fieldDec',
                'fiveSigmaDepth',
                'moonPhase',
                'subprogram']
        where=''
        if self.use_dbcut:
            where=" WHERE "
            for cut_var in self.db_cut:
                where+="("
                for cut in self.db_cut[cut_var]:
                    if cut_var == 'subprogram' or  cut_var=='filter':
                        cut_str=f"='{cut}'"
                    else:
                        cut_str=f"{cut}"
                    where+=f"{cut_var}{cut_str} OR "
                where=where[:-4]
                where+=") AND "
            where=where[:-5]
        obs_dic={}
        for k in keys:
            query = 'SELECT '+k+' FROM Summary'+where+';'
            values = dbf.execute(query)
            self.obs_dic[k] = np.array([a[0] for a in values])
        return

    def make_obs_table(self, epochs_selec):
        # Capture noise and filter
        mlim5 = self.obs_dic['fiveSigmaDepth'][epochs_selec]
        filter = self.obs_dic['filter'][epochs_selec].astype('U27')
        # Change band name to correpond with sncosmo bands
        if self.band_dic is not None:
            for i, f in enumerate(filter):
                filter[i] = self.band_dic[f]

        # Convert maglim to flux noise (ADU)
        skynoise = pw(10., 0.4 * (self.zp - mlim5)) / 5

        # Create obs table
        obs = Table({'time': self.obs_dic['expMJD'][epochs_selec],
                     'band': filter,
                     'gain': [self.gain] * np.sum(epochs_selec),
                     'skynoise': skynoise,
                     'zp': [self.zp] * np.sum(epochs_selec),
                     'zpsys': ['ab'] * np.sum(epochs_selec)})
        if not self.use_obs:
            obs['subprogram'] = self.obs_dic['subprogram'][epochs_selec]
        self.obs.append(obs)
        return

    def read_host_file(self):
        stime = time.time()
        with fits.open(self.host_file) as hostf:
            host_list = hostf[1].data[:]
        host_list['ra'] = host_list['ra'] + 2 * np.pi * (host_list['ra'] < 0)
        l = f'HOST FILE READ IN  {time.time() - stime:.1f} seconds'
        print(su.box_output(su.sep, l))
        print(su.box_output(su.sep, '------------'))
        host_list = host_list[host_list['redshift'] > self.z_range[0]]
        host_list = host_list[host_list['redshift'] < self.z_range[1]]
        return host_list

    def epochs_selection(self, ra, dec, z, t0):
        '''Select epochs that match the survey observations'''
        ModelMinT_obsfrm = self.sim_model.mintime() * (1 + z)
        ModelMaxT_obsfrm = self.sim_model.maxtime() * (1 + z)
        epochs_selec = abs(ra-self.obs_dic['fieldRA']) < self.ra_size # ra selection
        epochs_selec *= abs(dec-self.obs_dic['fieldDec']) < self.dec_size # dec selection
        # use to avoid 1e43 errors
        epochs_selec *= (self.obs_dic['fiveSigmaDepth'] > 0)
        epochs_selec *= (self.obs_dic['expMJD'] - t0 > ModelMinT_obsfrm) * \
            (self.obs_dic['expMJD'] - t0 < ModelMaxT_obsfrm)
        return epochs_selec

    def epochs_cut(self, epochs_selec, z, t0):
        '''Check if the SN pass the cuts'''
        for cut in self.nep_cut:
            cutMin_obsfrm, cutMax_obsfrm = cut[1] * (1 + z), cut[2] * (1 + z)
            test = epochs_selec * (self.obs_dic['expMJD'] - t0 > cutMin_obsfrm)
            test *= (self.obs_dic['expMJD'] - t0 < cutMax_obsfrm)
            if len(cut) == 4:
                test *= (np.vectorize(self.band_dic.get)
                         (self.obs_dic['filter']) == cut[3])
            if np.sum(test) < int(cut[0]):
                return False
        return True

    def gen_sn_rate(self, z):
        '''Give the rate of SN Ia given a volume'''
        rate = self.sn_rate * pw((1 + z),
                                 self.rate_pw)  # Rate in Nsn/Mpc^3/year
        shell_vol = 4 * np.pi / 3 * (pw(self.cosmo.comoving_distance(
            z + self.dz).value, 3) - pw(self.cosmo.comoving_distance(z).value, 3))
        time_rate = rate * shell_vol
        return time_rate

    def cadence_sim(self):
        '''Use a cadence file to produce SN according to a rate:
                1- Cut the zrange into shell (z,z+dz)
                2- Compute time rate for the shell r = r_v(z) * V SN/year where r_v is volume rate
                3- Generate the number of SN Ia in each shell with a Poisson's law
                4- Generate ra,dec for all the SN uniform on the sphere
                5- Generate t0 uniform between mintime and maxtime
                5- Generate z for in each shell uniform in the interval [z,z+dz]
                6- Apply observation and selection cut to SN
        '''
        if self.duration is None:
            self.duration = np.max(
                self.obs_dic['expMJD']) - np.min(self.obs_dic['expMJD'])
        # dz such as rv(z+dz)<(1+1e-2)rv(z)
        self.dz = 0.01 * (1 + self.z_range[1]) / np.max(1, self.rate_pw)
        if self.use_host:
            host_list = self.read_host_file()
            redshift_copy = np.sort(np.copy(host_list['redshift']))
            diff = redshift_copy[1:] - redshift_copy[:-1]
            # avoid dz < gap between redshift
            self.dz = np.max([self.dz, 2 * np.max(diff)])
        nstep = int((self.z_range[1] - self.z_range[0]) / self.dz)
        z_bins = np.linspace(self.z_range[0], self.z_range[1] - self.dz, nstep)
        time_rate = self.gen_sn_rate(z_bins)
        n_sn = np.random.default_rng(
            self.randseeds['nsn_seed']).poisson(
            self.duration * time_rate)
        self.n_sn_gen = np.sum(n_sn)
        t0_tmp = np.random.default_rng(
            self.randseeds['t0_seed']).uniform(
            np.min(
                self.obs_dic['expMJD']), np.max(
                self.obs_dic['expMJD']), size=self.n_sn_gen)

        self.ra = []
        self.dec = []
        self.sim_t0 = []
        self.zcos = []
        self.n_sn = 0

        if self.use_host:
            choice_seeds = np.random.default_rng(
                self.randseeds['coord_seed']).integers(
                low=1000, high=10000, size=len(n_sn))
            host_idx = []
            for z, n, rds in zip(z_bins, n_sn, choice_seeds):
                selec = (host_list['redshift'] >= z) * \
                    (host_list['redshift'] <= z + self.dz)
                host_in_shell = host_list[selec]
                self.selec = selec
                host_in_shell_idx = np.where(selec)[0]
                if len(host_in_shell) < n:
                    raise RuntimeError('No host in shell')
                host_tmp_idx = np.random.default_rng(rds).choice(
                    host_in_shell_idx, size=n, replace=False)
                host_tmp = host_list[host_tmp_idx]
                for idx, h, t0 in zip(host_tmp_idx, host_tmp, t0_tmp):
                    epochs_selec = self.epochs_selection(
                        h['ra'], h['dec'], h['redshift'], t0)
                    pass_cut = self.epochs_cut(epochs_selec, h['z'], t0)
                    if pass_cut:
                        self.host.append(h)
                        self.ra.append(h['ra'])
                        self.dec.append(h['dec'])
                        self.sim_t0.append(t0)
                        self.zcos.append(h['redshift'])
                        self.make_obs_table(epochs_selec)
                        host_idx.append(idx)
                        self.n_sn += 1
            self.host = host_list[host_idx]

        else:
            z_randseeds = np.random.default_rng(
                self.randseeds['z_seed']).integers(
                low=1000, high=10000, size=self.n_sn_gen)
            coord_seeds = np.random.default_rng(
                self.randseeds['coord_seed']).integers(
                low=1000, high=10000, size=2)

            ra_tmp, dec_tmp = self.gen_coord(coord_seeds, size=self.n_sn_gen)
            zcos_tmp = []

            for z, n, rds in zip(z_bins, n_sn, z_randseeds):
                zcos_tmp = np.concatenate(
                    (zcos_tmp, self.gen_redshift_cos(
                        low=z, high=z + self.dz, size=n, randseed=rds)))

            for ra, dec, zcos, t0 in zip(ra_tmp, dec_tmp, zcos_tmp, t0_tmp):
                epochs_selec = self.epochs_selection(ra, dec, zcos, t0)
                pass_cut = self.epochs_cut(epochs_selec, zcos, t0)
                if pass_cut:
                    self.ra.append(ra)
                    self.dec.append(dec)
                    self.sim_t0.append(t0)
                    self.zcos.append(zcos)
                    self.make_obs_table(epochs_selec)
                    self.n_sn += 1

        self.ra = np.asarray(self.ra)
        self.dec = np.asarray(self.dec)
        self.zcos = np.asarray(self.zcos)
        return

    def fix_nsn_sim(self):
        '''Use a cadence db file to produce obs :
                1- Generate SN ra,dec in cadence fields
                2- Generate SN t0 in the survey time
                3- For each t0,ra,dec select visits that match
                4- Capture the information (filter, noise) of these visits
                5- Create sncosmo obs Table
         '''
        t0_seeds = np.random.default_rng(
            self.randseeds['t0_seed']).integers(
            low=1000, high=100000, size=self.n_sn)

        if self.use_host:
            self.zcos = []
            host_list = self.read_host_file()

            h_use_idx = []
            choice_seeds = np.random.default_rng(
                self.randseeds['coord_seed']).integers(
                low=1000, high=10000, size=self.n_sn)
        else:
            ra_seeds, dec_seeds = np.random.default_rng(
                self.randseeds['coord_seed']).integers(
                low=1000, high=10000, size=(
                    2, self.n_sn))

        self.ra = []
        self.dec = []
        self.sim_t0 = []
        self.n_gen = 0

        for i in range(self.n_sn):
            compt = 0
            pass_cut = False

            while not pass_cut:
                self.n_gen += 1
                # Gen ra and dec
                if self.use_host:
                    h_idx = np.random.default_rng(
                        choice_seeds[i]).choice(
                        np.arange(
                            len(host_list)),
                        replace=False)
                    ra, dec, z = host_list[h_idx]['ra'], host_list[h_idx]['dec'], host_list[h_idx]['redshift']
                else:
                    ra, dec = self.gen_coord([ra_seeds[i], dec_seeds[i]])
                    z = self.zcos[i]

                # Gen t0
                t0 = np.random.default_rng(
                    t0_seeds[i]).uniform(
                    np.min(
                        self.obs_dic['expMJD']), np.max(
                        self.obs_dic['expMJD']))

                # epochs selection
                epochs_selec = self.epochs_selection(ra, dec, z, t0)

                # Cut on epochs
                pass_cut = self.epochs_cut(epochs_selec, z, t0)

                if not pass_cut:
                    if self.use_host:
                        choice_seeds[i] = np.random.default_rng(
                            choice_seeds[i]).integers(1000, 100000)
                    else:
                        ra_seeds[i] = np.random.default_rng(
                            ra_seeds[i]).integers(1000, 100000)
                        dec_seeds[i] = np.random.default_rng(
                            dec_seeds[i]).integers(1000, 100000)
                    t0_seeds[i] = np.random.default_rng(
                        t0_seeds[i]).integers(1000, 100000)

                if compt > len(self.obs_dic['expMJD'] * 2):
                    raise RuntimeError(
                        'Too many cuts required, reduces nep_cut')
                else:
                    compt += 1

            self.make_obs_table(epochs_selec)

            self.ra.append(ra)
            self.dec.append(dec)
            self.sim_t0.append(t0)
            if self.use_host:
                h_use_idx.append(h_idx)

        self.ra = np.asarray(self.ra)
        self.dec = np.asarray(self.dec)

        if self.use_host:
            self.host = host_list[h_use_idx]
            self.zcos = self.host['redshift']
        return

    def write_sim(self):
        '''Write the simulated lc in a fits file'''
        lc_hdu_list = []
        self.sn_id = []
        for i, tab in enumerate(self.sim_lc):
            tab.meta['vpec'] = self.vpec[i]
            tab.meta['zcos'] = self.zcos[i]
            tab.meta['zpec'] = self.zpec[i]
            tab.meta['z2cmb'] = self.z2cmb[i]
            tab.meta['zCMB'] = self.zCMB[i]
            tab.meta['ra'] = self.ra[i]
            tab.meta['dec'] = self.dec[i]
            tab.meta['sn_id'] = i
            tab.meta['mb'] = self.sim_mB[i]
            tab.meta['mu'] = self.sim_mu[i]
            tab.meta['msmear'] = self.mag_smear[i]
            self.sn_id.append(i)
            lc_hdu_list.append(fits.table_to_hdu(tab))
        sim_header = {'n_sn': self.n_sn,'alpha': self.alpha, 'beta': self.beta, 'M0': self.M0, 'SIG_M': self.sigmaM}
        hdu_list = fits.HDUList(
            [fits.PrimaryHDU(header=fits.Header(sim_header))] + lc_hdu_list)
        hdu_list.writeto(
            self.write_path +
            self.sim_name +
            '.fits',
            overwrite=True)
        self.sn_id = np.asarray(self.sn_id)
        #Export lc as pickle
        with open(self.write_path+self.sim_name+'_lcs.pkl','wb') as file:
            pickle.dump(self.sim_lc,file)
        return

    def fitter(self, id):
        '''Use sncosmo to fit sim lc'''
        try:
            res = su.snc_fit(self.sim_lc[id], self.fit_model)
        except BaseException:
            self.fit_res[id] = 'NaN'
            return
        self.fit_res[id] = res
        return

    def fit_lc(self, lc_id=None):
        '''Send the lc and model to fit to self.fitter'''
        if lc_id is None:
            for i in self.sn_id:
                # set the model's redshift.
                self.fit_model.set(z=self.sim_lc[i].meta['z'])
                self.fitter(i)
        else:
            self.fit_model.set(z=self.sim_lc[lc_id].meta['z'])
            self.fitter(lc_id)
        return

    def write_fit(self):

        sim_lc_meta = {'id': self.sn_id,
                       'ra': self.ra,
                       'dec': self.dec,
                       'vpec': self.vpec,
                       'zpec': self.zpec,
                       'z2cmb': self.z2cmb,
                       'zcos': self.zcos,
                       'zCMB': self.zCMB,
                       'zobs': self.zobs,
                       'sim_x0': self.sim_x0,
                       'sim_mb': self.sim_mB,
                       'sim_x1': self.sim_x1,
                       'sim_c': self.sim_c,
                       'sim_mu': self.sim_mu,
                       'sim_mag_smear': self.mag_smear
                       }

        sim_meta={'n_sn': self.n_sn, 'alpha': self.alpha, 'beta': self.beta, 'M0': self.M0, 'SIG_M': self.sigmaM}
        su.write_fit(sim_lc_meta,self.fit_res,self.write_path+self.sim_name+'_fit.fits',sim_meta=sim_meta)
        return

    def plot_lc(
            self,
            lc_id,
            zp=25.,
            mag=False,
            plot_sim=True,
            plot_fit=False,
            residuals=False):
        '''Ploting the ligth curve of number 'lc_id' sn'''
        if plot_sim:
            sim_model = self.sim_model
        else:
            sim_model = None

        if plot_fit:
            if self.fit_res[lc_id] == 'No_fit':
                raise ValueError("This lc wasn't fitted")
            fit_model = self.fit_res[lc_id][1]
            fit_cov = self.fit_res[lc_id][0]['covariance'][1:, 1:]
        else:
            fit_model = None
            fit_cov = None
        su.plot_lc(
            self.sim_lc[lc_id],
            zp=zp,
            mag=mag,
            sim_model=self.sim_model,
            fit_model=fit_model,
            fit_cov=fit_cov,
            residuals=residuals)
        return


class open_sim:
    def __init__(self, sim_file, SALT2_dir):
        '''Copy some function of snsim to allow to use sim file'''
        self.salt2_dir = SALT2_dir
        source = snc.SALT2Source(modeldir=self.salt2_dir)
        self.model = snc.Model(source=source)
        self.sim_lc = []
        self.meta={}
        with fits.open(sim_file) as sf:
            self.header=sf[0].header
            self.n_sn = sf[0].header['n_sn']
            meta=True
            for i,hdu in enumerate(sf[1:]):
                data = hdu.data
                tab = Table(data)
                tab.meta = hdu.header

                if meta:
                    meta=False
                    for k in tab.meta:
                        self.meta[k]=np.zeros(self.n_sn,dtype='object')
                for k in tab.meta:
                    self.meta[k][i]=tab.meta[k]

                self.sim_lc.append(tab)
        self.fit_res = np.asarray(['No_fit'] * self.n_sn, dtype='object')

        return

    def plot_lc(
            self,
            lc_id,
            zp=25.,
            mag=False,
            plot_sim=True,
            plot_fit=False,
            residuals=False):
        '''Ploting the ligth curve of number 'lc_id' sn'''
        if plot_sim:
            sim_model = self.model
        else:
            sim_model = None

        if plot_fit:
            if self.fit_res[lc_id] == 'No_fit':
                raise ValueError("This lc wasn't fitted")
            fit_model = self.fit_res[lc_id][1]
            fit_cov = self.fit_res[lc_id][0]['covariance'][1:, 1:]
        else:
            fit_model = None
            fit_cov = None
        su.plot_lc(
            self.sim_lc[lc_id],
            zp=zp,
            mag=mag,
            sim_model=self.model,
            fit_model=fit_model,
            fit_cov=fit_cov,
            residuals=residuals)
        return

    def fitter(self, id):
        '''Use sncosmo to fit sim lc'''
        try:
            res = su.snc_fit(self.sim_lc[id], self.model)
        except BaseException:
            self.fit_res[id] = 'NaN'
            return
        self.fit_res[id] = res
        return

    def fit_lc(self, lc_id=None):
        '''Send the lc and model to fit to self.fitter'''
        if lc_id is None:
            for i in range(self.n_sn):
                # set the model's redshift.
                self.model.set(z=self.sim_lc[i].meta['z'])
                self.fitter(i)
        else:
            self.model.set(z=self.sim_lc[lc_id].meta['z'])
            self.fitter(lc_id)
        return

    def write_fit(self,w_path):
        sim_meta_keys=['n_sn', 'alpha', 'beta', 'M0', 'SIG_M']
        sim_meta={}

        for k in sim_meta_keys:
            sim_meta[k]=self.header[k]

        sim_lc_meta = {'id': [],
                           'ra': [],
                           'dec': [],
                           'vpec': [],
                           'zpec': [],
                           'z2cmb': [],
                           'zcos': [],
                           'zCMB': [],
                           'zobs': [],
                           'sim_x0': [],
                           'sim_mb': [],
                           'sim_x1': [],
                           'sim_c': [],
                           'sim_mu': [],
                           'sim_mag_smear': []
                           }
        trad_dic = {'id': 'sn_id',
                    'ra': 'ra',
                    'dec': 'dec',
                    'vpec': 'vpec',
                    'zpec': 'zpec',
                    'z2cmb': 'z2cmb',
                    'zcos': 'zcos',
                    'zCMB': 'zcmb',
                    'zobs': 'z',
                    'sim_x0': 'x0',
                    'sim_mb': 'mb',
                    'sim_x1': 'x1',
                    'sim_c': 'c',
                    'sim_mu': 'mu' ,
                    'sim_mag_smear':'msmear'
                   }

        for lc in self.sim_lc:
            for k in trad_dic:
                sim_lc_meta[k].append(lc.meta[trad_dic[k]])

        su.write_fit(sim_lc_meta,self.fit_res,w_path+'_fit.fits',sim_meta=sim_meta)
        return
