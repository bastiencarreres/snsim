import numpy as np
import sqlite3
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from . import sim_utils as su

class SNGen:
    '''The SN generator class, take sim parameters as init, and can be call to simulate a given number of SN object
    Data structure :
    SNGen
    ├── snc_sim_model
    ├── model_par
    │   ├── M0 # The absolute magnitude of SN Ia
    │   ├── sigCOH # Coherent scattering value
    │   ├── time_range # time range of the simulation
    │   └── sn_model_parameter... # parameters as x_1, c for SALT2
    └── cosmo_dic
        ├── CMB
        │   ├── v_cmb
        │   ├── ra_cmb
        │   └── dec_cmb
        └── cosmo_par
            ├── HO
            └── Om
    '''
    def __init__(self,snc_sim_model,sim_par,host=None):
        self.sim_model = snc_sim_model
        self.model_par = sim_par['model_par']
        self.model_keys = ['M0']
        self.model_keys += self.init_model()
        self.cosmo_dic = sim_par['cosmo_dic']
        self.host = host

    def isSalt(self):
        if self.sim_model.source.name in ['salt2','salt3']:
            return True
        return False

    def init_model(self):
        if self.isSalt:
            model_keys = ['alpha','beta']
        return model_keys

    @property
    def sigCOH(self):
        return self.model_par['sigCOH']

    def __call__(self,n_sn,z_range,rand_seed):
        rand_seeds = np.random.default_rng(rand_seed).integers(low=1000, high=100000,size=7)

        t0 = self.gen_peak_time(n_sn,rand_seeds[0])
        ra, dec = self.gen_coord(n_sn,rand_seeds[1])
        zcos = self.gen_zcos(n_sn,z_range,rand_seeds[2])
        vpec = self.gen_vpec(n_sn,rand_seeds[3])
        mag_smear = self.gen_coh_scatter(n_sn,rand_seeds[4])
        noise_rand_seed = self.gen_noise_rand_seed(n_sn,rand_seeds[5])
        model_par_sncosmo = self.gen_sncosmo_param(n_sn,rand_seeds[6:8])

        sn_par = [{'zcos': z,
                   'sim_t0': t,
                   'ra': r,
                   'dec': d,
                   'vpec': v,
                   'mag_smear': ms
                   } for z,t,r,d,v,ms in zip(zcos,t0,ra,dec,vpec,mag_smear)]

        model_default = {}
        for k in self.model_keys:
            model_default[k] = self.model_par[k]

        model_par_list = [{**model_default, 'sncosmo': mpsn, 'noise_rand_seed': rs } for mpsn, rs in zip(model_par_sncosmo, noise_rand_seed)]
        print(model_par_list)
        SN_list = [SN(snp,self.cosmo_dic,self.sim_model,mp) for snp,mp in zip(sn_par,model_par_list)]
        return SN_list

    def gen_peak_time(self,n_sn,rand_seed):
        t0 = np.random.default_rng(rand_seed).uniform(*self.model_par['time_range'],size=n_sn)
        return t0

    def gen_coord(self,n_sn,rand_seed):
        coord_seed = np.random.default_rng(rand_seed).integers(low=1000,high=100000,size=2)
        ra = np.random.default_rng(coord_seed[0]).uniform(
            low=0, high=2 * np.pi, size=n_sn)
        dec_uni = np.random.default_rng(coord_seed[1]).random(size=n_sn)
        dec = np.arcsin(2 * dec_uni - 1)
        return ra, dec

    def gen_zcos(self,n_sn,z_range, rand_seed):
        '''Function to get zcos, to be updated'''
        zcos = np.random.default_rng(rand_seed).uniform(
        low=z_range[0], high=z_range[1], size=n_sn)
        return zcos

    def gen_sncosmo_param(self,n_sn,rand_seed):
        if self.isSalt:
            sim_x1, sim_c = self.gen_SALT_par(n_sn,rand_seed[0])
            model_par_sncosmo = [{'x1': x1, 'c': c} for x1, c in zip(sim_x1,sim_c)]

        if 'G10_' in self.sim_model.effect_names:
            seeds = np.random.default_rng(rand_seed[1]).integers(low=1000, high=100000,size=n_sn)
            for par,s in zip(model_par_sncosmo,seeds):
                model_par_sncosmo['G10_RndS'] = s

        elif 'C11_' in self.sim_model.effect_names:
            seeds = np.random.default_rng(rand_seed[1]).integers(low=1000, high=100000,size=n_sn)
            for par,s in zip(model_par_sncosmo,seeds):
                par['C11_RndS'] = s

        return model_par_sncosmo

    def gen_SALT_par(self,n_sn,rand_seed):
        ''' Generate x1 and c for the SALT2 or SALT3 model'''
        x1_seed, c_seed = np.random.default_rng(rand_seed).integers(low=1000, high=100000,size=2)
        sim_x1 = np.random.default_rng(x1_seed).normal(
            loc=self.model_par['x1_distrib'][0],
            scale=self.model_par['x1_distrib'][1],
            size=n_sn)
        sim_c = np.random.default_rng(c_seed).normal(
            loc=self.model_par['c_distrib'][0], scale=self.model_par['c_distrib'][1], size=n_sn)
        return sim_x1, sim_c

    def gen_vpec(self,n_sn,rand_seed):
        if self.host == None:
            vpec = np.random.default_rng(rand_seed).normal(
                loc=self.model_par['vpec_distrib'][0],
                scale=self.model_par['vpec_distrib'][1],
                size=n_sn)

        else:
            vpec = host.vpec

        return vpec

    def gen_coh_scatter(self,n_sn,rand_seed):
        ''' Generate coherent intrinsic scattering '''
        mag_smear = np.random.default_rng(rand_seed).normal(loc=0, scale=self.sigCOH, size=n_sn)
        return mag_smear

    def gen_noise_rand_seed(self,n_sn,rand_seed):
        return np.random.default_rng(rand_seed).integers(low=1000, high=100000,size=n_sn)


class SN:
    '''Supernova object
    Data structure :
    SN
    ├── _sn_par (base attributes that allow to compute secondary attributes)
    │   ├── sim_t0
    │   ├── zcos
    │   ├── ra
    │   ├── dec
    │   ├── vpec
    │   └── mag_smear
    ├── cosmo_dic
    │   ├── CMB
    │   │   ├── ra_cmb
    │   │   ├── dec_cmb
    │   │   └── v_cmb
    │   └── cosmo_par
    │       ├── H0
    │       └── Om0
    ├── model_par
    │   ├── sncosmo (contains the variable that sncosmo needs to compute flux)
    │   └── other parameters (depends on sncosmo model source)
    ├── sim_model (sn_cosmo model)
    ├── _epochs
    ├── sim_lc
    └── _fit_model

    '''

    def __init__(self,sn_par,cosmo_dic,sim_model,model_par):
        self.sim_model = model.__copy__()
        self._sn_par = sn_par
        self.cosmo_dic = cosmo_dic
        self.model_par = model_par
        self.init_model_par()
        self._epochs = None
        self.sim_lc = None
        self._fit_model = None
        return

    @property
    def sim_t0(self):
        return self._sn_par['sim_t0']

    @property
    def vpec(self):
        return self._sn_par['vpec']

    @property
    def zcos(self):
        return self._sn_par['zcos']

    @property
    def coord(self):
        return self._sn_par['ra'], self._sn_par['dec']

    @property
    def mag_smear(self):
        return self._sn_par['mag_smear']

    @property
    def zpec(self):
        return self.vpec/su.c_light_kms

    @property
    def zCMB(self):
        return (1+self.zcos)*(1+self.zpec) - 1.

    @property
    def z2cmb(self):
        return self._sn_par['z2cmb']

    @property
    def z(self):
        return (1+self.zcos)*(1+self.zpec)*(1+self.z2cmb) - 1.

    @property
    def sim_mb(self):
        return self._sn_par['sim_mb']

    @property
    def epochs(self):
         return self._epochs

    @epochs.setter
    def epochs(self,ep_dic):
        self._epochs = ep_dic

    @property
    def z2cmb(self):
        ra_cmb = self.cosmo_dic['CMB']['ra_cmb']
        dec_cmb = self.cosmo_dic['CMB']['dec_cmb']
        v_cmb = self.cosmo_dic['CMB']['v_cmb']

        ra,dec = self.coord
        # use ra dec to simulate the effect of our motion
        coordfk5 = SkyCoord(ra * u.rad,
                            dec * u.rad,
                            frame='fk5')  # coord in fk5 frame

        galac_coord = coordfk5.transform_to('galactic')
        ra_gal = galac_coord.l.rad - 2 * np.pi * \
        np.sign(galac_coord.l.rad) * (abs(galac_coord.l.rad) > np.pi)
        dec_gal = galac_coord.b.rad

        ss = np.sin(dec_gal) * np.sin(dec_cmb * np.pi / 180)
        ccc = np.cos(dec_gal) * np.cos(dec_cmb * np.pi / \
                         180) * np.cos(ra_gal - ra_cmb * np.pi / 180)
        return (1 - v_cmb * (ss + ccc) / su.c_light_kms) - 1.

    @property
    def sim_mu(self):
        ''' Generate x0/mB parameters for SALT2 '''
        cosmo = FlatLambdaCDM(**self.cosmo_dic['cosmo_par'])
        return 5 * np.log10((1 + self.zcos) * (1 + self.z2cmb) * pw(
            (1 + self.zpec), 2) * cosmo.comoving_distance(self.zcos).value) + 25


    def init_model_par(self):
        ''' Init the SN magnitude in restframe Bessell B band'''
        M0 = self.model_par['M0']
        if self.sim_model.source.name in ['salt2', 'salt3']:
            # Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 +
            # beta*c : scattering due to color and stretch} + {intrinsic smearing}
            alpha = self.model_par['alpha']
            beta = self.model_par['beta']
            x1 = self.model_par['sncosmo']['x1']
            c = self.model_par['sncosmo']['c']
            mb = self.sim_mu + M0 - alpha * \
               x1 + beta * c+ self.mag_smear

            x0 = mB_to_x0(mb)

            setattr(SN,'sim_x1',x1)
            setattr(SN,'sim_c', c)
            setattr(SN,'sim_mb', mb)
            setattr(SN,'sim_x0',x0)

            self.model_par['sncosmo']['x0'] = x0

        elif self.sim_model.source.name == 'snoopy':
            #TODO
            return

    def pass_cuts(self,nep_cut):
        if self.epochs == None:
            return  False
        else:
            for cut in nep_cut:
                cutMin_obsfrm, cutMax_obsfrm = cut[1] * (1 + self.z), cut[2] * (1 + self.z)
                test = epochs_selec * (self.epochs['expMJD'] - t0 > cutMin_obsfrm)
                test *= (self.epochs['expMJD'] - self.t0 < cutMax_obsfrm)
                if len(cut) == 4:
                    test *= (self.epochs['filter'] == cut[3])
                if np.sum(test) < int(cut[0]):
                    return False
            return True

    def gen_lc(self,add_keys={}):
        ''' Generate simulated flux '''
        params = {**{'z': self.z,'t0': self.sim_t0}, **self.model_par['sncosmo']}
        self.sim_lc = snc.realize_lcs(self.epochs, self.sim_model, [params], scatter=False)[0]
        rs = self.model_par['noise_rand_seed']
        self.sim_lc['flux'] = np.random.default_rng(rs).normal(loc=self.sim_lc['flux'], scale=self.sim_lc['fluxerr'])

        for k in add_keys:
            self.sim_lc[k] = obs[k]
        self.reformat_sim_table()
        return

    def reformat_sim_table(self):
        for k in self.sim_lc.meta.copy():
            if k != 'z':
                self.sim_lc.meta['sim_'+k] = self.sim_lc.meta.pop(k)
        self.sim_lc.meta['vpec'] = self.vpec
        self.sim_lc.meta['zcos'] = self.zcos
        self.sim_lc.meta['zpec'] = self.zpec
        self.sim_lc.meta['z2cmb'] = self.z2cmb
        self.sim_lc.meta['zCMB'] = self.zCMB
        self.sim_lc.meta['ra'] = self.coord[0]
        self.sim_lc.meta['dec'] = self.coord[1]
        self.sim_lc.meta['sn_id'] =
        self.sim_lc.meta['sim_mb'] = self.sim_mb
        self.sim_lc.meta['sim_mu'] = self.sim_mu
        self.sim_lc.meta['m_smear'] = self.mag_smear
        return


class ObsTable:
    '''Survey observation class
    Data Strucure:
    ObsTable
    ├── db_file
    ├── _survey_prop
    │   ├── ra_size
    │   ├── dec_size
    │   ├── gain
    │   └── zp
    ├── band_dic
    ├── db_cuts
    └── add_keys
    '''
    def __init__(self,db_file,survey_prop,band_dic,db_cuts=None,add_keys=[]):
        self._survey_prop = survey_prop
        self.db_cuts = db_cuts
        self.db_file = db_file
        self.band_dic = band_dic
        self.add_keys = add_keys
        self.obs_dic={}
        self.extract_from_db()
        return

    @property
    def field_size(self):
        return self._survey_prop['ra_size'],self._survey_prop['dec_size']

    @property
    def gain(self):
        return self._survey_prop['gain']

    @property
    def zp(self):
        if 'zp' in self._survey_prop:
            return self._survey_prop['zp']
        else:
            return 'zp_in_obs'

    @property
    def mintime(self):
        return np.min(self.obs_dic['expMJD'])

    @property
    def maxtime(self):
        return np.max(self.obs_dic['expMJD'])

    def extract_from_db(self):
        '''Read db file and extract relevant information'''
        dbf = sqlite3.connect(self.db_file)

        keys = ['expMJD',
                'filter',
                'fieldRA',
                'fieldDec',
                'fiveSigmaDepth']+self.add_keys

        where=''
        if self.db_cuts != None:
            where=" WHERE "
            for cut_var in self.db_cuts:
                where+="("
                for cut in self.db_cuts[cut_var]:
                    cut_str=f"{cut}"
                    where+=f"{cut_var}{cut_str} OR "
                where=where[:-4]
                where+=") AND "
            where=where[:-5]
        for k in keys:
            query = 'SELECT '+k+' FROM Summary'+where+';'
            values = dbf.execute(query)
            self.obs_dic[k] = np.array([a[0] for a in values])
        return

    def epochs_selection(self, SN):
        '''Select epochs that match the survey observations'''
        ModelMinT_obsfrm = SN.sim_model.mintime() * (1 + SN.z)
        ModelMaxT_obsfrm = SN.sim_model.maxtime() * (1 + SN.z)
        ra,dec = SN.coord
        # time selection
        epochs_selec = (self.obs_dic['expMJD'] - SN.sim_t0 > ModelMinT_obsfrm) * \
            (self.obs_dic['expMJD'] - SN.sim_t0 < ModelMaxT_obsfrm)
        # use to avoid 1e43 errors
        epochs_selec *= (self.obs_dic['fiveSigmaDepth'] > 0)
        # Find the index of the field that pass time cut
        epochs_selec_idx = np.where(epochs_selec)
        # Compute the coord of the SN in the rest frame of each field
        ra_size, dec_size = self.field_size
        ra_field_frame, dec_field_frame = change_sph_frame(ra,dec,self.obs_dic['fieldRA'][epochs_selec],self.obs_dic['fieldDec'][epochs_selec])
        epochs_selec[epochs_selec_idx] *= abs(ra_field_frame) < ra_size/2 # ra selection
        epochs_selec[epochs_selec_idx] *= abs(dec_field_frame) < dec_size/2 # dec selection
        if np.sum(epochs_selec) == 0:
            return None
        return self.make_obs_table(epochs_selec)


    def make_obs_table(self,epochs_selec):
        # Capture noise and filter
        mlim5 = self.obs_dic['fiveSigmaDepth'][epochs_selec]
        filter = self.obs_dic['filter'][epochs_selec].astype('U27')

        # Change band name to correpond with sncosmo bands -> CHANGE EMPLACEMENT
        if self.band_dic is not None:
            for i, f in enumerate(filter):
                filter[i] = self.band_dic[f]

        if self.zp != 'zp_in_obs':
            zp = [self.zp] * np.sum(epochs_selec)
        elif isinstance(zp,(int, float)):
            zp = self.obs_dic['zp'][epochs_selec]
        else:
            raise ValueError("zp is not define")

        # Convert maglim to flux noise (ADU)
        skynoise = pw(10., 0.4 * (self.zp - mlim5)) / 5

        # Create obs table
        obs = Table({'time': self.obs_dic['expMJD'][epochs_selec],
                      'band': filter,
                      'gain': [self.gain] * np.sum(epochs_selec),
                      'skynoise': skynoise,
                      'zp': zp,
                      'zpsys': ['ab'] * np.sum(epochs_selec)})

        for k in self.add_keys:
            obs[k] = self.obs_dic[k][epochs_selec]
        return obs
