"""Main module of the simulaiton package."""

import time
import yaml
import numpy as np
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
    |     rate: rate of SN/Mpc^3/year or 'ptf19'  # Optional, default=3e-5
    |     rate_pw: rate = rate*(1+z)^rate_pw  # Optional, default=0
    |     M0: SN ABSOLUT MAGNITUDE
    |     mag_sct: SN INTRINSIC COHERENT SCATTERING
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
    |
    | vpec_dist:
    |     mean_vpec: MEAN SN PECULIAR VELOCITY
    |     sig_vpec: SIGMA VPEC
    | host: (Optional)
    |     host_file: 'PATH/TO/HOSTFILE'
    |     distrib: 'as_sn', 'as_host' or 'mass_weight'  # Optional, default = 'as_sn'
    |     key_dic: {'column_name': 'new_column_name', etc}  # Optional, to change columns names
    | dipole:  # Optional, add a dipole as dM = A + B * cos(theta)
    |     coord: [RA, Dec]  # Direction of the dipole
    |     A: A_parameter
    |     B: B_parameter
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

        # Check if there is a db_file
        if 'survey_config' not in self.config:
            raise KeyError("Set a survey_file -> type help(sn_sim) to print the syntax")

        # Check if the sfdmap need to be download
        if 'mw_dust' in self.config:
            dst_ut.check_files_and_dowload()

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
            self._host = sh.SnHost(self.config['host'], self.z_range)
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
        for object_name in __GEN_DIC__:
            if object_name in self.config:
                # -- Get which generator correspond to which transient in snsim.generators
                gen_class = getattr(generators, __GEN_DIC__[object_name])
                self._generators.append(gen_class(self.config[object_name],
                                                  self.cmb,
                                                  self.cosmology,
                                                  vpec_dist=self.vpec_dist,
                                                  host=self.host,
                                                  mw_dust=mw_dust,
                                                  dipole=dipole,
                                                  survey_footprint=self.survey.fields.footprint))
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
            if isinstance(nep_cut, (int)):
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

    def peak_time_range(self, trange_model):
        """Get the time range for simulate obj peak.

        Returns
        -------
        tuple(float, float)
            Min and max time for obj peak generation.
        """
        min_peak_time = self.survey.start_end_days[0] - trange_model[1] * (1 + self.z_range[1])
        max_peak_time = self.survey.start_end_days[1] + abs(trange_model[0]) * (1 + self.z_range[1])
        return min_peak_time, max_peak_time

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
        4- Write LC to fits/pkl file(s)

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

        rate_str = "Rate r = {0:.2e} * (1 + z)^{1} /Mpc^3/year "

        # -- Compute time range, rate and zcdf for each of the selected obj.
        for use_rate, gen in zip(self._use_rate, self.generators):
            gen.print_config()

            # -- Set the time range with time edges effects
            peak_time_range = self.peak_time_range(gen.snc_model_time)
            gen.time_range = (peak_time_range[0].mjd, peak_time_range[1].mjd)

            if use_rate:
                rate_str = rate_str.format(gen.rate_law[0], gen.rate_law[1]) + "\n"
                compute_z_cdf = True
            else:
                print(f"\nGenerate {gen._params['force_n']} SN Ia")
                if self.host is not None and self.host.config['distrib'].lower() != 'as_sn':
                    rate_str = 'Redshift distribution computed '
                    if self.host.config['distrib'] == 'as_host':
                        rate_str += 'as host redshift distribution\n\n'
                    elif self.host.config['distrib'] == 'mass_weight':
                        rate_str += 'as mass weighted host redshift distribution\n\n'
                    compute_z_cdf = False
                else:
                    rate_str = rate_str.format(gen.rate_law[0], gen.rate_law[1])
                    rate_str += ' (only for redshifts simulation)\n\n'
                    compute_z_cdf = True

            if compute_z_cdf:
                gen.compute_zcdf(self.z_range)

            print('\n' + rate_str +
                  "Peak mintime : "
                  f"{peak_time_range[0].mjd:.2f} MJD / {peak_time_range[0].iso}\n"
                  "Peak maxtime : "
                  f"{peak_time_range[1].mjd:.2f} MJD / {peak_time_range[1].iso} ")

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

        # -- Simulation for each of the selected obj.
        for use_rate, seed, gen in zip(self._use_rate, seed_list, self.generators):
            sim_time = time.time()
            if use_rate:
                lcs_list = self._cadence_sim(np.random.default_rng(seed), gen, Obj_ID)
            else:
                lcs_list = self._fix_nsn_sim(np.random.default_rng(seed), gen, Obj_ID)

            self._samples.append(SimSample.fromDFlist(self.sim_name + '_' + gen._object_type,
                                                      lcs_list,
                                                      {**gen._get_header(),
                                                       **self._get_cosmo_header()},
                                                      model_dir=None,
                                                      dir_path=self.config['data']['write_path']))

            print(f'{len(lcs_list)} {gen._object_type} lcs generated'
                  f' in {time.time() - sim_time:.1f} seconds')
            write_time = time.time()
            self._samples[-1]._write_sim(self.config['data']['write_path'],
                                         self.config['data']['write_format'])

            print(f'Sim file write in {time.time() - write_time:.1f} seconds')

        print('\n-----------------------------------------------------------\n')

        print('OUTPUT FILE(S) : ')
        formats = np.atleast_1d(self.config['data']['write_format'])
        for f in formats:
            print('- '
                  + self.config['data']['write_path']
                  + self.sim_name + '_' + gen._object_type
                  + '.'
                  + f)

    def _cadence_sim(self, rand_gen, generator, Obj_ID=0):
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
        lcs = []
        if 'duration_for_rate' in self.config['sim_par']:
            duration = self.config['sim_par']['duration_for_rate']
        else:
            duration = generator.time_range[1] - generator.time_range[0]

        n_obj = self._gen_n_sn(rand_gen, generator._z_time_rate[1],
                               duration, area=self.survey.fields._tot_area)

        # -- Generate n base param
        param_tmp = generator.gen_astrobj_par(n_obj, rand_gen.integers(1000, 1e6))

        # -- Select observations that pass all the cuts
        epochs, parmask = self.survey.epochs_selection(param_tmp.to_records(index=False),
                                                       (generator.sim_model.mintime(),
                                                        generator.sim_model.maxtime()),
                                                       self.nep_cut)
        # -- Keep the parameters of selected lcs
        param_tmp = param_tmp[parmask]
        param_tmp.reset_index(inplace=True)

        # -- Generate the object
        obj_list = generator(np.sum(parmask),
                             rand_gen.integers(1000, 1e6),
                             astrobj_par=param_tmp)

        for ID, obj  in zip(epochs.index.unique('ID'), obj_list):
            obj.epochs = epochs.loc[[ID]]
            obj.gen_flux(rand_gen)
            obj.ID = ID
            lcs.append(obj.sim_lc)
        return lcs

    def _fix_nsn_sim(self, rand_gen, generator, Obj_ID=0):
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
        lcs = []
        raise_trigger = 0
        n_to_sim = generator._params['force_n']
        while len(lcs) < generator._params['force_n']:

            # -- Generate n base param
            param_tmp = generator.gen_astrobj_par(n_to_sim, rand_gen.integers(1000, 1e6))

            # -- Select observations that pass all the cuts
            epochs, parmask = self.survey.epochs_selection(param_tmp.to_records(index=False),
                                                           (generator.sim_model.mintime(),
                                                            generator.sim_model.maxtime()),
                                                           self.nep_cut, IDmin=len(lcs))
            if epochs is None:
                raise_trigger += 1
                if raise_trigger > 2 * len(self.survey.obs_table['expMJD']):
                    raise RuntimeError('Cuts are too stricts')
                continue

            # -- Keep the parameters of selected lcs
            param_tmp = param_tmp[parmask]
            param_tmp.reset_index(inplace=True)

            # -- Generate the object
            obj_list = generator(np.sum(parmask),
                                 rand_gen.integers(1000, 1e6),
                                 astrobj_par=param_tmp)

            for ID, obj in zip(epochs.index.unique('ID'), obj_list):
                obj.epochs = epochs.loc[[ID]]
                obj.gen_flux(rand_gen)
                obj.ID = ID
                lcs.append(obj.sim_lc)

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
