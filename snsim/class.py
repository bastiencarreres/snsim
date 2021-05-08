
import numpy as np
from astropy.cosmology import FlatLambdaCDM

class SN:
    '''SN object
    Data structure :
    SN
    ├── sn_par
    │   ├── sim_t0
    │   ├── z
    │   ├── zcos
    │   ├── zCMB
    │   ├── z2cmb
    │   ├── zpec
    │   ├── vpec
    │   ├── sim_mu
    │   ├── sim_mb
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
    │   ├── sncosmo
    │   └── other parameters (depends on sncosmo model source)
    ├── sim_model (sn_cosmo model)
    ├── epochs
    ├── sim_lc
    └── fit_model
    '''
    __slots__ = ['sn_par','cosmo_dic','sim_model','model_par']

    def __init__(self,sn_par,cosmo_dic,sim_model,model_par):
        self.sn_par = sn_par
        self.sn_par['z2cmb'] = self.init_z2cmb(cosmo_dic)
        self.sn_par['sim_mu'] = self.init_sim_mu(cosmo_dic)
        self.model_par = model_par
        self.sim_model = model.__copy__()
        self._epochs = None
        self.sim_lc = None
        self._fit_model = None
        return

    @property
    def sim_t0(self):
        return self.sn_par['sim_t0']

    @property
    def zcos(self):
        return self.sn_par['zcos']

    @property
    def coord(self):
        return self.sn_par['ra'], self.sn_par['dec']

    @property
    def mag_smear(self):
        return self.sn_par['mag_smear']

    @property
    def zpec(self):
        return self.vpec/su.c_light_kms

    @property
    def zCMB(self):
        return (1+self.zcos)*(1+self.zpec) - 1.

    @property
    def z2cmb(self):
        return self.sn_par['z2cmb']

    @property
    def z(self):
        return (1+self.zcos)*(1+self.zpec)*(1+self.z2cmb) - 1.

    @property
    def sim_mu(self):
        return self.sn_par['sim_mu']

    @property
    def sim_mb(self):
        return self.sn_par['sim_mb']

    @property
    def epochs(self):
         return self._epochs

    @epochs.setter
    def epochs(self,ep_dic):
        self._epochs = ep_dic


    def init_z2cmb(self,cosmo_par):
        ra_cmb = cosmo_dic['CMB']['ra_cmb']
        dec_cmb = cosmo_par['CMB']['dec_smb']
        v_cmb = cosmo_par['CMB']['v_cmb']
        # use ra dec to simulate the effect of our motion
        coordfk5 = SkyCoord(self.ra * u.rad,
                            self.dec * u.rad,
                            frame='fk5')  # coord in fk5 frame

        galac_coord = coordfk5.transform_to('galactic')
        ra_gal = galac_coord.l.rad - 2 * np.pi * \
        np.sign(galac_coord.l.rad) * (abs(galac_coord.l.rad) > np.pi)
        dec_gal = galac_coord.b.rad

        ss = np.sin(dec_gal) * np.sin(dec_cmb * np.pi / 180)
        ccc = np.cos(dec_gal) * np.cos(dec_cmb * np.pi / \
                         180) * np.cos(ra_gal - ra_cmb * np.pi / 180)
        z2cmb = (1 - v_cmb * (ss + ccc) / su.c_light_kms) - 1.
        return z2cmb

    def init_sim_mu(self,cosmo_dic):
        ''' Generate x0/mB parameters for SALT2 '''
        cosmo = FlatLambdaCDM(**cosmo_dic['cosmo_par'])
        self.sn_par['sim_mu'] = 5 * np.log10((1 + self.zcos) * (1 + self.z2cmb) * pw(
            (1 + self.zpec), 2) * self.cosmo.comoving_distance(self.zcos).value) + 25

        return

    def init_sim_mb(self,model_par):
        ''' Init the SN magnitude in restframe Bessell B band'''
        M0 = self.model_par['M0']

        if self.sim_model.source.name in ['salt2', 'salt3']:
            # Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 +
            # beta*c : scattering due to color and stretch} + {intrinsic smearing}
            alpha = self.model_par['alpha']
            beta = self.model_par['beta']
            sim_mb = self.sim_mu + M0 - alpha * \
                self.sim_x1 + beta * self.sim_c + self.mag_smear
            self.model_par['sncosmo']['sim_x0'] = su.mB_to_x0(sim_mB)
            return self.sim_mB

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
                    test *= (self.epochs['filter']) == cut[3])
                if np.sum(test) < int(cut[0]):
                    return False
            return True

    def gen_flux(self,add_keys={}):
        ''' Generate simulated flux '''
        obs_table = self.make_obs_table(self.epochs)
        params = {**{'z': self.z,'t0': self.t0}, **modeldir}
        self.sim_lc = snc.realize_lcs(obs_table, self.sim_model, [params], scatter=False)[0]

        self.sim_lc['flux'] = np.random.default_rng(s).normal(
                loc=lc['flux'], scale=lc['fluxerr'])

        for k in add_keys:
            self.sim_lc[k] = obs[k]


class ObsTable:
    __slots__ = ['obs_dic','ra_size','dec_size']

    def __init__(self,db_file,db_cuts,add_keys,ra_size,dec_size):
        self.ra_size = ra_size
        self.dec_size = dec_size
        self.obs_dic={}
        self.extract_from_db()
        return

    def extract_from_db(self,db_cuts = None):
        '''Read db file and extract relevant information'''
        dbf = sqlite3.connect(db_file)

        keys = ['expMJD',
                'filter',
                'fieldRA',
                'fieldDec',
                'fiveSigmaDepth']+add_keys

        where=''
        if db_cuts != None:
            where=" WHERE "
            for cut_var in db_cuts:
                where+="("
                for cut in db_cuts[cut_var]:
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
        ModelMinT_obsfrm = self.sim_model.mintime() * (1 + z)
        ModelMaxT_obsfrm = self.sim_model.maxtime() * (1 + z)

        # time selection
        epochs_selec = (self.obs_dic['expMJD'] - t0 > ModelMinT_obsfrm) * \
            (self.obs_dic['expMJD'] - t0 < ModelMaxT_obsfrm)
        # use to avoid 1e43 errors
        epochs_selec *= (self.obs_dic['fiveSigmaDepth'] > 0)
        # Find the index of the field that pass time cut
        epochs_selec_idx = np.where(epochs_selec)
        # Compute the coord of the SN in the rest frame of each field
        ra_field_frame, dec_field_frame = su.change_sph_frame(ra,dec,self.obs_dic['fieldRA'][epochs_selec],self.obs_dic['fieldDec'][epochs_selec])
        epochs_selec[epochs_selec_idx] *= abs(ra_field_frame) < self.ra_size/2 # ra selection
        epochs_selec[epochs_selec_idx] *= abs(dec_field_frame) < self.dec_size/2 # dec selection
        if np.sum(epochs_selec) == 0:
            return None
        return self.make_obs_table(epochs_selec)


        def make_obs_table(self,epochs_selec):
            # Capture noise and filter
            mlim5 = self.epochs['fiveSigmaDepth'][epochs_selec]
            filter = self.epochs['filter'][epochs_selec].astype('U27')

            # Change band name to correpond with sncosmo bands -> CHANGE EMPLACEMENT
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

            for k in self.add_keys:
                obs[k] = self.obs_dic[k][epochs_selec]

            return obs
