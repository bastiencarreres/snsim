"""Main module of the simulaiton package."""

import time
import yaml
import numpy as np
import dask
import astropy.units as aunits
from . import utils as ut
from . import generators
from . import survey_host as sh
from .constants import SN_SIM_PRINT, VCMB, L_CMB, B_CMB
from . import dust_utils as dst_ut

from .generators import __GEN_DIC__
from .sample import SimSample


class Simulator:
    """Simulation class using a config file config.yml.

    Parameters
    ----------
    param_dic : dict / str
        The configuration yaml file / The dictionnary containing all simulation parameters.

    Notes
    -----
    yaml file format :

    | data :
    |     write_path: '/PATH/TO/OUTPUT'
    |     sim_name: 'NAME OF SIMULATION'
    |     write_format: 'format' or ['format1','format2']
    | survey_config:
    |     survey_file: '/PATH/TO/FILE'
    |     band_dic: {'r':'ztfr','g':'ztfg','i':'ztfi'}  # Optional -> if bandname in
    |  survey_file doesn't match sncosmo name
    |     key_dic: {'column_name': 'new_column_name', etc}  # Optional, to change columns names
    |     add_data: ['keys1', 'keys2', ...] add db file keys to metadata
    |     survey_cut: {'key1': ['conditon1','conditon2',...], 'key2': ['conditon1']}
    |     start_day: MJD NUMBER or 'YYYY-MM-DD'  # Optional, default given by survey file
    |     duration: SURVEY DURATION (DAYS)  # Optional, default given by survey file
    |     zp: INSTRUMENTAL ZEROPOINT  # Optional, default given by survey file
    |     sig_zp: ZEROPOINT ERROR  # Optional, default given by survey file
    |     sig_psf: GAUSSIAN PSF SIGMA  # Otional, default given by survey file as FWHMeff
    |     noise_key: [key, type] type can be 'mlim5' or 'skysigADU'
    |     ra_size: RA FIELD SIZE
    |     dec_size: DEC FIELD SIZE
    |     gain: CCD GAIN e-/ADU (Optional, default given by survey file)
    |     sub_field: ['sub_field_file', 'sub_field_key']
    |     fake_skynoise: [SIGMA_VALUE, 'add' or 'replace']  # Optional, default is 0
    | sim_par:
    |     randseed: RANDSEED TO REPRODUCE SIMULATION  # Optional
    |     z_range: [ZMIN, ZMAX]
    |     nep_cut: [[nep_min1,Tmin,Tmax],[nep_min2,Tmin2,Tmax2,'filter1'],...] EP CUTS
    |     duration_for_rate: FAKE DURATION ONLY USE TO GENERATE N OBJ  # Optional
    | mw_dust:
    |     model: MOD_NAME
    |     rv: Rv # Optional, default Rv = 3.1
    | snia_gen:
    |     n_sn: NUMBER OF SN TO GENERATE  # Optional
    |     rate: rate of SN/Mpc^3/year # Optional, default=3e-5
    |     M0: SN ABSOLUT MAGNITUDE
    |     sigM: SN INTRINSIC COHERENT SCATTERING
    |     sct_model: 'G10','C11_i' USE WAVELENGHT DEP MODEL FOR SN INT SCATTERING
    |     model_config:
    |         model_name: 'THE MODEL NAME'  Example : 'salt2'
    |         model_dir: '/PATH/TO/SALT/MODEL'
    |         alpha: STRETCH CORRECTION = alpha*x1
    |         beta: COLOR CORRECTION = -beta*c
    |         dist_x1: [MEAN X1, SIGMA X1], [MEAN X1, SIGMA_X1_LOW, SIGMA_X1_HIGH] or 'N21'
    |         dist_c: [MEAN C, SIGMA C] or [SIGMA_C_LOW, SIGMA_C_HIGH]
    | cosmology:
    |     Om0: MATTER DENSITY
    |     H0: HUBBLE CONSTANT
    | cmb:
    |     v_cmb: OUR PECULIAR VELOCITY  # Optional, default = 620 km/s
    |     l_cmb: GAL L OF CMB DIPOLE  # Optional, default = 271.0
    |     b_cmb: GAL B OF CMB DIPOLE  # Optional, default = 29.6
    | vpec_dist:
    |     mean_vpec: MEAN SN PECULIAR VELOCITY
    |     sig_vpec: SIGMA VPEC
    | host: (Optional)
    |     host_file: 'PATH/TO/HOSTFILE'
    |     distrib: 'rate' or 'random'  # Optional, default = 'rate'
    |     key_dic: {'column_name': 'new_column_name', etc}  # Optional, to change columns names
    | dipole:  # Optional, add a dipole as dM = A + B * cos(theta)
    |     coord: [RA, Dec]  # Direction of the dipole
    |     A: A_parameter
    |     B: B_parameter
    | dask: # Optional for using dask parallelization
    |     use: True or False
    |     nworkers: NUMBER OF WORKERS # used to adjust work distribution
    """

    def __init__(self, param_dic, print_config=False):
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

        if 'dask' in self.config:
            if 'nworkers' not in self.config['dask']:
                self.config['dask']['nworkers'] = 10
        else:
            self.config['dask']['use'] = False

        # Check if there is a db_file
        if 'survey_config' not in self.config:
            raise KeyError("Set a survey_file -> type help(sn_sim) to print the syntax")

        # Check if the sfdmap need to be download
        if 'mw_dust' in self.config:
            dst_ut.check_files_and_download()

        self._sample = None
        self._random_seed = None

        # -- Init cosmological model
        self._cosmology = ut.set_cosmo(self.config['cosmology'])

        # -- Init SurveyObs object
        self._survey = sh.SurveyObs(self.config['survey_config'])

        # -- Init vpec_dist
        if 'vpec_dist' in self.config:
            self._vpec_dist = self.config['vpec_dist']
        else:
            self._vpec_dist = None

        # -- Init host object
        if 'host' in self.config:
            self._host = sh.SnHost(self.config['host'], z_range=self.z_range,
                                   geometry=self.survey._envelope)
        else:
            self._host = None

        # -- Init mw dust
        if 'mw_dust' in self.config:
            mw_dust = self.config['mw_dust']
        else:
            mw_dust = None

        # -- Init dipole
        if 'dipole' in self.config:
            dipole = self.config['dipole']
        else:
            dipole = None

        # Init the cuts on lightcurves
        self._nep_cut = self._init_nep_cuts()

        # -- Init generators for each transients
        self._use_rate = []
        self._generators = []

        time_range = (self.survey.start_end_days[0].mjd, self.survey.start_end_days[1].mjd)

        for object_name, object_genclass in __GEN_DIC__.items():
            if object_name in self.config:
                # -- Get which generator correspond to which transient in snsim.generators
                gen_class = getattr(generators, object_genclass)
                self._generators.append(gen_class(self.config[object_name],
                                                  self.cmb,
                                                  self.cosmology,
                                                  time_range,
                                                  z_range=self.z_range,
                                                  peak_out_trange=True,
                                                  vpec_dist=self.vpec_dist,
                                                  host=self.host,
                                                  mw_dust=mw_dust,
                                                  dipole=dipole,
                                                  geometry=self.survey._envelope))
                # -- Cadence sim or n fixed
                if 'force_n' in self.config[object_name]:
                    self._use_rate.append(False)
                else:
                    self._use_rate.append(True)

        if print_config:
            print('PARAMETERS USED IN SIMULATION\n')
            ut.print_dic(self.config)

        # -- Init samples attributes (to store simulated obj)
        self._samples = None

    def _init_nep_cuts(self):
        """Init nep cut on transients.

        Returns
        -------
        numpy.ndarray(int, float, float, str)
            Numpy array containing cuts.

        Notes
        -----
        Format of a cut is [# of ep, mintime, maxtime, filter]

        """
        # -- Set default mintime, maxtime (restframe)
        snc_mintime = -20
        snc_maxtime = 50
        cut_list = []
        if 'nep_cut' in self.config['sim_par']:
            nep_cut = self.config['sim_par']['nep_cut']
            if isinstance(nep_cut, (int, np.integer)):
                cut_list.append((nep_cut, snc_mintime, snc_maxtime, 'any'))
            elif isinstance(nep_cut, (list)):
                for i, cut in enumerate(nep_cut):
                    if len(cut) < 3:
                        cut_list.append((cut[0], snc_mintime, snc_mintime, 'any'))
                    elif len(cut) < 4:
                        cut_list.append((cut[0], cut[1], cut[2], 'any'))
                    else:
                        cut_list.append((cut[0], cut[1], cut[2], cut[3]))
        else:
            cut_list = [(1, snc_mintime, snc_maxtime, 'any')]
        dt = [('nep', np.int8), ('mintime', np.int8), ('maxtime', np.int8), ('band', np.str_, 8)]
        return np.asarray(cut_list, dtype=dt)

    def _gen_n_sn(self, rand_gen, z_shell_time_rate, duration_in_days, area=4 * np.pi):
        """Generate the number of obj with Poisson law.

        Parameters
        ----------
        rand_seed : int
            The random seed for the random generator.

        Returns
        -------
        int
            Number of obj to simulate.

        """
        nsn = duration_in_days / 365.25 * area / (4 * np.pi) * np.sum(z_shell_time_rate)
        nsn = int(np.round(nsn))
        return rand_gen.poisson(nsn)

    def _get_cosmo_header(self):
        """Return the header for cosmology model used."""
        if 'name' in self.config['cosmology']:
            return {'cosmod_name': self.config['cosmology']['name']}
        else:
            return self.config['cosmology']

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
        4- Write LCs to parquet/pkl file(s)

        """
        print(SN_SIM_PRINT)

        print('-----------------------------------------------------------\n')

        print(f"SIM NAME : {self.sim_name}\n"
              f"CONFIG FILE : {self._yml_path}\n"
              f"SIM WRITE DIRECTORY : {self.config['data']['write_path']}\n"
              f"SIMULATION RANDSEED : {self.randseed}")

        if 'host_file' in self.config:
            print(f"HOST FILE : {self.config['host_file']}")

        print('\n-----------------------------------------------------------\n')

        print(self.survey)

        print('\n-----------------------------------------------------------\n')

        # -- Compute time range, rate and zcdf for each of the selected obj.
        for use_rate, gen in zip(self._use_rate, self.generators):
            
            rate_str = f"\nRate {gen._rate_expr} /Mpc^3/year "
            print(gen)
            if not use_rate:
                rate_str += ' (only for redshifts simulation)\n'
        
            print(rate_str)
            
        print('\n-----------------------------------------------------------\n')

        if 'mw_dust' in self.config:
            print("Use mw dust model : "
                  f"{self.config['mw_dust']['model']} with RV = {self.config['mw_dust']['rv']}")

            print('\n-----------------------------------------------------------\n')

        print("Ligthcurves cuts :")

        for cut in self.nep_cut:
            print_cut = f'- At least {cut[0]} epochs between {cut[1]} and {cut[2]} rest-frame phase'
            if len(cut) == 4:
                print_cut += f' in {cut[3]} band'
            print(print_cut)

        print('\n-----------------------------------------------------------\n')

        # -- Create the random generator object with the rand seed
        rand_gen = np.random.default_rng(self.randseed)
        seed_list = rand_gen.integers(1000, 1e6, size=len(self.generators))

        # -- Change the samples attribute to store obj, init ID
        self._samples = []
        Obj_ID = 0
        file_str=''

        # -- Simulation for each of the selected obj.
        for use_rate, seed, gen in zip(self._use_rate, seed_list, self.generators):
            sim_time = time.time()
            if use_rate:
                lcs_list = self._cadence_sim(np.random.default_rng(seed), gen, Obj_ID)
            else:
                lcs_list = self._fix_nsn_sim(np.random.default_rng(seed), gen, Obj_ID)
            
            self._samples.append(SimSample.fromDFlist(self.sim_name + '_' + gen._object_type,
                                                      lcs_list,
                                                      {'seed': seed,
                                                       **gen._get_header(),
                                                       'cosmo': self._get_cosmo_header()},
                                                      model_dir=None,
                                                      dir_path=self.config['data']['write_path']))

            print(f'{len(lcs_list)} {gen._object_type} lcs generated'
                  f' in {time.time() - sim_time:.1f} seconds')
            write_time = time.time()
            self._samples[-1]._write_sim(self.config['data']['write_path'],
                                         self.config['data']['write_format'])

            print(f'Sim file write in {time.time() - write_time:.1f} seconds')

            formats = np.atleast_1d(self.config['data']['write_format'])
            for f in formats:
                file_str += '- '+ self.config['data']['write_path']+ self.sim_name + '_' + gen._object_type + '.'+ f + '\n'

        print('\n-----------------------------------------------------------\n')

        print('OUTPUT FILE(S) : ')
        print(file_str)

    def _sim_lcs(self, generator, n_obj, Obj_ID=0, seed=None):
        """Simulate AstrObj lcs.

        Parameters
        ----------
        generator : snsim.generator
            The parameter generator class
        n_obj : int
            The nummber of object to generate
        Obj_ID : int, optional
           The first ID of AstrObj, by default 0
        seed : int, optional
            The random seed to generate parameters, by default None

        Returns
        -------
        list(pandas.Dataframe)
            List of the AstrObj LCs

        """
        if seed is None:
            seed = np.random.randint(1e3, 1e6)

        rand_gen = np.random.default_rng(seed)

        # -- Init lcs list
        lcs = []

        # -- Generate n base param
        param_tmp = generator.gen_astrobj_par(n_obj, rand_gen.integers(1000, 1e6), 
                                              min_max_t=True)

        # -- Set up obj parameters
        model_t_range = (generator.snc_model_time[0], generator.snc_model_time[1])

        # -- Select observations that pass all the cuts
        epochs, params = self.survey.get_observations(param_tmp,
                                                      phase_cut=model_t_range,
                                                      nep_cut=self.nep_cut,
                                                      IDmin=Obj_ID,
                                                      use_dask=self.config['dask']['use'],
                                                      npartitions=self.config['dask']['nworkers'])
        if params is None:
            raise RuntimeError('None of the object pass the cuts...')
        
        # -- Generate the object
        obj_list = generator(rand_seed=rand_gen.integers(1e3, 1e6),
                             astrobj_par=params)

        # -- TO DO: dask it when understanding the random pickel-sncosmo error

        # if self.config['dask']['use']:
        #     it_edges = np.linspace(0, len(obj_list) + 1,
        #                            int(np.ceil(len(obj_list) / (10 * self.config['dask']['nworkers']) + 1)),
        #                            dtype=int)

        #     for imin, imax in zip(it_edges[:-1], it_edges[1:]):
        #         lcs += dask.compute([dask.delayed(obj).gen_flux(epochs.loc[[obj.ID]])
        #                              for obj in obj_list[imin:imax]])[0]
        # else:
        lcs = [obj.gen_flux(epochs.loc[[obj.ID]]) for obj in obj_list]
        return lcs

    def _cadence_sim(self, rand_gen, generator, Obj_ID=0):
        """Simulate a number of AstrObj according to poisson law.

        Parameters
        ----------
        rand_gen : numpy.random.default_rng
            Numpy random generator.
        shell_time_rate : numpy.ndarray
            An array that contains sn time rate in each shell.

        Returns
        -------
        list(pandas.Dataframe)
            List of the AstrObj LCs

        Notes
        -----
        Simulation routine:
            1- Cut the zrange into shell (z, z + dz)
            2- Compute time rate for the shell r = r_v(z) * V(z) AstrObj / year where r_v is the volume rate
            3- Generate the number of SN Ia in each shell with a Poisson's law
            4- Generate ra, dec for all the SN uniform on the sphere
            5- Generate t0 uniform between mintime and maxtime
            6- Generate z following the rate distribution
            7- Apply observation and selection cuts to SN
            8- Genertate fluxes
        """
        # -- Generate the number of SN
        if 'duration_for_rate' in self.config['sim_par']:
            duration = self.config['sim_par']['duration_for_rate']
        else:
            duration = generator.time_range[1] - generator.time_range[0]

        n_obj = self._gen_n_sn(rand_gen, generator._z_time_rate[1],
                               duration, area=self.survey._envelope_area)

        lcs = self._sim_lcs(generator, n_obj,
                            Obj_ID=Obj_ID, seed=rand_gen.integers(1e3, 1e6))

        return lcs

    def _fix_nsn_sim(self, rand_gen, generator, Obj_ID=0):
        """Simulate a fixed number of AstrObj.

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
        lcs = []
        raise_trigger = 0
        n_to_sim = generator._params['force_n']
        while len(lcs) < generator._params['force_n']:
            lcs += self._sim_lcs(generator, n_to_sim,
                                 Obj_ID=len(lcs), seed=rand_gen.integers(1e3, 1e6))

            # -- Arbitrary cut to stop the simulation if no SN are geenrated
            if n_to_sim == generator._params['force_n'] - len(lcs):
                raise_trigger += 1
                if raise_trigger > 2 * len(self.survey.obs_table['expMJD']):
                    raise RuntimeError('Cuts are too stricts')
                continue
           
            n_to_sim = generator._params['force_n'] - len(lcs)

        return lcs

    def plot_ra_dec(self, idx, plot_vpec=False, plot_fields=False, **kwarg):
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
        if plot_fields:
            field_dic = self.survey.fields._dic
            field_size = self.survey.fields.size
        else:
            field_dic = None
            field_size = None

        self.samples[idx].plot_ra_dec(plot_vpec=plot_vpec,
                                      field_dic=field_dic,
                                      field_size=field_size,
                                      **kwarg)

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
        return self._vpec_dist

    @property
    def cmb(self):
        """Get cmb parameters."""
        cmb_dic = {'v_cmb': VCMB, 'l_cmb': L_CMB, 'b_cmb': B_CMB}
        if 'cmb' in self.config:
            if 'v_cmb' in self.config['cmb']:
                cmb_dic['v_cmb'] = self.config['cmb']['v_cmb']
            if 'l_cmb' in self.config['cmb']:
                cmb_dic['l_cmb'] = self.config['cmb']['l_cmb']
            if 'b_cmb' in self.config['cmb']:
                cmb_dic['b_cmb'] = self.config['cmb']['b_cmb']
        return cmb_dic

    @property
    def samples(self):
        """Get the list of simulated sn."""
        return self._samples

    @property
    def cosmology(self):
        """Get astropy cosmological model used in simulation."""
        return self._cosmology

    @property
    def survey(self):
        """Get the SurveyObs object of the simulation."""
        return self._survey

    @property
    def generators(self):
        """Get the SNGen object of the simulation."""
        return self._generators

    @property
    def host(self):
        """Get the SnHost object of the simulation."""
        return self._host

    @property
    def randseed(self):
        """Get primary random seed of the simulation."""
        if 'randseed' in self.config['sim_par']:
            return int(self.config['sim_par']['randseed'])
        elif self._random_seed is None:
            self._random_seed = np.random.randint(low=1000, high=100000)
        return self._random_seed

    @property
    def z_range(self):
        """Get z_range."""
        return self.config['sim_par']['z_range']

    @property
    def nep_cut(self):
        """Get the list of epochs cuts."""
        return self._nep_cut
