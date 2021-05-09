import numpy as np
from numpy import power as pw
from astropy.table import Table
from astropy.io import fits
import yaml
import os
import pickle
import time
from . import sim_utils as su
from . import scatter as sct

class SnSim:
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
        | db file doesn't correpond to those in sncosmo registery)                         |
        |     write_format: 'format' or ['format1','format2'] # Optional default pkl, fits |
        | db_config: #(Optional -> use obs_file)                                           |
        |     dbfile_path: '/PATH/TO/FILE'                                                 |
        |     add_keys: ['keys1', 'keys2', ...] add db file keys to metadata               |
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
        |     M0: SN ABSOLUT MAGNITUDE                                                     |
        |     mag_smear: SN INTRINSIC SMEARING                                             |
        |     smear_mod: 'G10','C11_i' USE WAVELENGHT DEP MODEL FOR SN INT SCATTERING      |
        | cosmology:                                                                       |
        |     Om: MATTER DENSITY                                                           |
        |     H0: HUBBLE CONSTANT                                                          |
        |     v_cmb: OUR PECULIAR VELOCITY #(Optional, default = 369.82 km/s)              |
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

    # Load param dict from a yaml or by using launch_script.py
        if isinstance(param_dic, dict):
            self.sim_cfg = param_dic
            self._yml_path = param_dic['yaml_path']

        elif isinstance(param_dic, str):
            self._yml_path = param_dic
            with open(self._yml_path, "r") as f:
                self.sim_cfg = yaml.safe_load(f)

        # Check if there is a db_file
        if 'db_config' not in self.sim_cfg:
            raise RuntimeError("Set a db_file -> type help(sn_sim) to print the syntax")

        # cadence sim or n fixed
        if 'n_sn' in self.sim_cfg['sn_gen']:
            self.n_sn = int(self.sim_cfg['sn_gen']['n_sn'])
            self.use_rate = False
            self.duration = None
        else:
            self.use_rate = True


    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #--------------- vpec_gen section -----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        if 'host_file' in self.sim_cfg['vpec_gen']:
            self.use_host = True
            self.host_file = self.sim_cfg['vpec_gen']['host_file']
        else:
            self.use_host = False

    # Minimal nbr of epochs in LC


    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #------------ Randseed Initialisation -------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#
        self._random_seed = None

        self._obs = ObsTable(**self.obs_parameters)
        self._generator = SNGen(self.sim_par)

    @property
    def sim_name(self):
        return self.sim_cfg['data']['sim_name']

    @property
    def write_format(self):
        if 'write_format' in self.sim_cfg['data_cfg']['write_format']:
            write_format = self.sim_cfg['data_cfg']['write_format']
            if isinstance(write_format,str):
                if write_format not in ['fits','pkl']:
                    raise ValueError('write_format avaible are fits and pkl')
                write_format = [write_format]

            elif isinstance(self.data_cfg['write_format'],list):
                for format in self.data_cfg['write_format']:
                    if format not in ['fits','pkl']:
                        raise ValueError('write_format avaible are fits and pkl')
        else:
            write_format =  ['fits', 'pkl']
        return write_format

    @property
    def survey_prop(self):
        dic = {'ra_size': np.radians(self.sim_cfg['db_config']['ra_size']),
                'dec_size': np.radians(self.sim_cfg['db_config']['dec_size']),
                'gain': self.sim_cfg['db_config']['gain']
                }
        # zeropoint
        if 'zp' in self.sim_cfg['db_config']:
            dic['zp'] = self.sim_cfg['db_config']['zp']
        return dic

    @property
    def obs_parameters(self):
        params = {'db_file': self.sim_cfg['db_config']['dbfile_path'], 'survey_prop': self.survey_prop}
        # Band dic : band_name_obs/db_file -> band_name_sncosmo
        if 'band_dic' in  self.sim_cfg['db_config']:
            band_dic = self.sim_cfg['db_config']['band_dic']
        else:
            band_dic = None
        params['band_dic'] = band_dic
        # Additionnal data
        if 'add_keys' in  self.sim_cfg['db_config']:
            add_keys = self.db_cfg['add_keys']
        else:
            add_keys = []
        params['add_keys'] = add_keys

        # Cut on db_file
        if 'db_cut' in  self.sim_cfg['db_config']:
            db_cut = self.sim_cfg['db_config']['db_cut']
        else:
            db_cut = None
        params['db_cut'] = db_cut
        return params

    @property
    def snc_model_par(self):
        if 'salt_gen' in self.sim_cfg:
            model_tag = 'salt'
            model_dir = self.sim_cfg['salt_gen']['salt_dir']
            version = self.sim_cfg['salt_gen']['version']
            if version not in [2, 3]:
                raise ValueError("Support SALT version = 2 or 3")

        params = {'model_tag': model_tag,
                  'model_dir': model_dir,
                  'version': version}

        if 'smear_mod' in self.sim_cfg['sn_gen']:
            params['smear_mod'] = self.sim_cfg['sn_gen']['smear_mod']
        return params

    @property
    def sn_model_par(self):
        params = {'M0': self.sim_cfg['sn_gen']['M0'],
                  'time_range': [self.obs.mintime,self.obs.maxtime],
                  'mag_smear': self.sim_cfg['sn_gen']['mag_smear']}

        if 'salt_gen' in self.sim_cfg:
            params['alpha'] = self.sim_cfg['salt_gen']['alpha']
            params['beta'] = self.sim_cfg['salt_gen']['beta']
            params['x1_distrib'] = [self.sim_cfg['salt_gen']['mean_x1'],self.sim_cfg['salt_gen']['sig_x1']]
            params['c_distrib'] = [self.sim_cfg['salt_gen']['mean_c'],self.sim_cfg['salt_gen']['sig_c']]

        if 'host_file' not in self.sim_cfg['vpec_gen']:
            params['vpec_distrib'] = [self.sim_cfg['vpec_gen']['mean_vpec'], self.sim_cfg['vpec_gen']['sig_vpec']]
        return params

    @property
    def cosmo(self):
        return {'H0': self.sim_cfg['cosmology']['H0'], 'Om0': self.sim_cfg['cosmology']['Om']}

    @property
    def cmb(self):
        params = {}
        if 'v_cmb' in self.sim_cfg['cosmology']:
            params['v_cmb'] = self.sim_cfg['cosmology']['v_cmb']
        else :
            params['v_cmb'] = 369.82
        params['dec_cmb'] = 48.253
        params['ra_cmb'] = 266.81
        return params

    @property
    def sim_par(self):
        return {'snc_model_par': self.snc_model_par,
                'cosmo': self.cosmo,
                'cmb': self.cmb,
                'sn_model_par': self.sn_model_par}

    @property
    def obs(self):
        if self._obs._survey_prop != self.survey_prop:
            self._obs = ObsTable(**self.obs_parameters)
        return self._obs

    @property
    def generator(self):
        not_same = (self._generator._sim_par != self.sim_par)
        if not_same:
            self._generator = SNGen(self.sim_par)
        return self._generator

    @property
    def fit_model(self):
        print('fit_model')
        if 'salt_gen' in self.sim_cfg:
            self.salt_dir = self.sim_cfg['salt_gen']['salt_dir']
            if self.salt_gen['version'] == 2:
                source = snc.SALT2Source(modeldir=self.salt_dir)
            elif self.salt_gen['version'] == 3:
                source = snc.SALT3Source(modeldir=self.salt_dir)
            else :
                raise RuntimeError("Support SALT version = 2 or 3")
        return snc.Model(source=source)

    @property
    def nep_cut(self):
        snc_mintime, snc_maxtime = self.generator.snc_model_time
        if 'nep_cut' in self.sim_cfg['sn_gen']:
            nep_cut = self.sim_cfg['sn_gen']['nep_cut']
            if isinstance(nep_cut, (int)):
                nep_cut = [
                    (nep_cut,
                     snc_mintime,
                     snc_maxtime)]
            elif isinstance(nep_cut, (list)):
                for i in range(len(self.nep_cut)):
                    if len(self.nep_cut[i]) < 3:
                        nep_cut[i].append(snc_mintime)
                        nep_cut[i].append(snc_maxtime)
        else:
            nep_cut = [(1, snc_mintime, snc_maxtime)]
        return nep_cut

    @property
    def rand_seed(self):
        if 'randseed' in self.sim_cfg['sn_gen']:
            return int(self.sim_cfg['sn_gen']['randseed'])
        elif self._random_seed is None:
            self._random_seed = np.random.randint(low=1000, high=100000)
        return self._random_seed

    @property
    def z_range(self):
        return self.sim_cfg['sn_gen']['z_range']


    @property
    def survey_duration(self):
        if 'duration' not in self.sim_cfg['sn_gen']:
            duration = (self.obs.mintime() - self.obs.maxtime())/365.25
        else:
            duration = self.sim_cfg['sn_gen']['duration']
        return duration

    @property
    def z_span(self):
        z_min, z_max = self.z_range
        rate_pw = self.sim_cfg['sn_gen']['rate_pw']
        dz = (z_max-z_min)/(100*(1+rate_pw*z_max))
        n_bins = int((z_max - z_min) / dz)
        z_bins = np.linspace(z_min, z_max - dz, n_bins)
        return  {'z_bins': z_bins, 'dz': dz, 'n_bins': n_bins }

    @property
    def sn_rate_z0(self):
        if 'sn_rate' and 'rate_pw' in  self.sim_cfg['sn_gen'] :
            return float(self.sim_cfg['sn_gen']['sn_rate']), self.sim_cfg['sn_gen']['rate_pw']
        else:
            return 3e-5, 0

    def sn_rate(self,z):
        rate_z0, pw = self.sn_rate_z0
        return  rate_z0 * (1 + z)**pw

    def gen_sn_rate(self, z):
        '''Give the rate of SN Ia given a volume'''

        cosmo = FlatLambdaCDM(**self.cosmo)
        rate = self.sn_rate(z)# Rate in Nsn/Mpc^3/year
        shell_vol = 4 * np.pi / 3 * (pw(cosmo.comoving_distance(
            z + self.z_span['dz']).value, 3) - pw(cosmo.comoving_distance(z).value, 3))
        time_rate = rate * shell_vol
        return time_rate

    def gen_n_sn(self,rand_seed):
        z_bins = self.z_span['z_bins']
        time_rate = self.gen_sn_rate(z_bins)
        return np.random.default_rng(rand_seed).poisson(self.survey_duration * time_rate)

    def simulate(self):
        '''Simulation routine :
        1- READ OBS/DB FILE
        2- GEN REDSHIFT AND SALT2 PARAM
        3- GEN LC FLUX WITH sncosmo
        4- WRITE LC TO A FITS FILE
        '''

        #print(su.sn_sim_print)
        print('-------------------------------------------')
        print(f'SIM NAME : {self.sim_name}')
        print(f'CONFIG FILE : {self._yml_path}')
        print(f"OBS FILE : {self.obs_parameters['db_file']}")
        print(f"SIM WRITE DIRECTORY : {self.sim_cfg['data']['write_path']}")
        print(f'SIMULATION RANDSEED : {self.rand_seed}')
        print(f'-------------------------------------------\n')

        if self.use_rate:
            if self.survey_duration is None:
                duration_str = f'Survey duration is given by cadence file'
            else:
                duration_str = f'Survey duration is {self.survey_duration} year(s)'

            print(f"Generate with a rate of r_v = {self.sn_rate_z0[0]}*(1+z)^{self.sn_rate_z0[1]} SN/Mpc^3/year")
            print(duration_str + '\n')
        else:
            print(f"Generate {self.n_sn} SN Ia")

        if self.obs_parameters['db_cut'] is not None:
            for k,v in self.obs_parameters['db_cut'].items():
                conditions_str=''
                for cond in v:
                    conditions_str+=str(cond)+' OR '
                conditions_str=conditions_str[:-4]
                print(f'Select {k}: '+conditions_str)
        else:
            print('No db cut')
        print('\n')

        print("SN ligthcurve cuts :")

        for cut in self.nep_cut:
            print_cut = f'- At least {cut[0]} epochs between {cut[1]} and {cut[2]}'
            if len(cut)==4:
                print_cut+=f' in {cut[3]} band'
            print(print_cut)
        print('\n')

        self.use_host = False
        if self.use_host:
            self.host = []
            with fits.open(self.host_file) as hduf:
                for hdu in hduf:
                    self.host.append(hdu.data)

        start_time = time.time()
        ############################
        #sep2 = su.box_output(su.sep, '------------')
        line = f'OBS FILE read in {time.time()-start_time:.1f} seconds'
        #print(su.sep)
        #print(su.box_output(su.sep, line))
        #print(sep2)

        sim_time = time.time()
        ############################
        if self.use_rate:
            self.cadence_sim()
        else:
            self.fix_nsn_sim()
        ############################
        l = f'{len(self.sn_list)} SN lcs generated in {time.time() - sim_time:.1f} seconds'
        print(l)
        #print(su.box_output(su.sep, l))
        #print(sep2)
        '''
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
        print('\n OUTPUT FILES : ')
        for format in self.write_format:
            print('- '+self.write_path+self.sim_name+'.'+format)
        '''
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


    def cadence_sim(self):
        '''Use a cadence file to produce SN according to a rate:
                1- Cut the zrange into shell (z,z+dz)
                2- Compute time rate for the shell r = r_v(z) * V SN/year where r_v is the volume rate
                3- Generate the number of SN Ia in each shell with a Poisson's law
                4- Generate ra,dec for all the SN uniform on the sphere
                5- Generate t0 uniform between mintime and maxtime
                5- Generate z for in each shell uniform in the interval [z,z+dz]
                6- Apply observation and selection cuts to SN
        '''

        n_sn_seed, sn_gen_seed = np.random.default_rng(self.rand_seed).integers(low=1000, high=100000, size=2)

        if self.use_host:
            host_list = self.read_host_file()
            redshift_copy = np.sort(np.copy(host_list['redshift']))
            diff = redshift_copy[1:] - redshift_copy[:-1]
            # avoid dz < gap between redshift
            self.dz = np.max([self.dz, 2 * np.max(diff)])

        n_sn = self.gen_n_sn(n_sn_seed)

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
            sn_bins_seed = np.random.default_rng(sn_gen_seed).integers(low=1000, high=100000, size=np.sum(n_sn))
            self.sn_list = []
            for n, z, rs in zip(n_sn, self.z_span['z_bins'], sn_bins_seed):
                sn_list_tmp = self.generator(n,[z,z+self.z_span['dz']],rs)
                for sn in sn_list_tmp:
                    sn.epochs = self.obs.epochs_selection(sn)
                    if sn.pass_cut(self.nep_cut):
                        sn.gen_flux()
                        self.sn_list.append(sn)
        return None

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
        self.sn_id = []
        if 'fits' in self.write_format:
            lc_hdu_list = []
            if 'fits' in self.write_format:
                lc_hdu_list.append(fits.table_to_hdu(tab))

        self.sn_id = np.asarray(self.sn_id)
        if 'fits' in self.write_format:
            sim_header = {'n_sn': self.n_sn,'alpha': self.alpha, 'beta': self.beta, 'M0': self.M0, 'SIG_M': self.sigmaM}
            hdu_list = fits.HDUList(
                    [fits.PrimaryHDU(header=fits.Header(sim_header))] + lc_hdu_list)
            hdu_list.writeto(
                    self.write_path +
                    self.sim_name +
                    '.fits',
                    overwrite=True)

        #Export lcs as pickle
        if 'pkl' in self.write_format:
            with open(self.write_path+self.sim_name+'_lcs.pkl','wb') as file:
                pickle.dump(self.sim_lc,file)
        return
