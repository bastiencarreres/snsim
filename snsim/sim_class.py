import numpy as np
import sqlite3
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from . import sim_utils as su

class SnHost:
    def __init__(self,host_file,z_range = None):
        self.z_range = z_range
        self.host_file = host_file
        self.host_list = self.read_host_file()
        self._max_dz = None

    @property
    def max_dz(self):
        if self._max_dz is None:
            redshift_copy = np.sort(np.copy(self.host_list['redshift']))
            diff = redshift_copy[1:] - redshift_copy[:-1]
            self._max_dz = np.max(diff)
        return self._max_dz

    def read_host_file(self):
        #stime = time.time()
        with fits.open(self.host_file) as hostf:
            host_list = hostf[1].data[:]
        host_list['ra'] = host_list['ra'] + 2 * np.pi * (host_list['ra'] < 0)
        #l = f'HOST FILE READ IN  {time.time() - stime:.1f} seconds'
        #print(su.box_output(su.sep, l))
        #print(su.box_output(su.sep, '------------'))
        if self.z_range is not None:
            return self.host_in_range(host_list,self.z_range)
        else:
            return host_list

    @staticmethod
    def host_in_range(host,z_range):
        selec = host['redshift'] > z_range[0]
        selec *= host['redshift'] < z_range[1]
        return host[selec]

    def random_host(self, n_host, z_range, random_seed):
        if z_range[0] < self.z_range[0] or z_range[1] > self.z_range[1]:
            raise ValueError(f'z_range must be between {self.z_range[0]} and {self.z_range[1]}')
        elif z_range[0] > z_range[1]:
            raise ValueError(f'z_range[0] must be < to z_range[1]')
        host_available = self.host_in_range(self.host_list, z_range)
        host_choice = np.random.default_rng(random_seed).choice(host_available, size=n_host, replace=False)
        if len(host_choice) < n_host:
            raise RuntimeError('Not enough host in the shell')
        return host_choice


class ObsTable:
    def __init__(self,db_file,survey_prop,band_dic=None,db_cut=None,add_keys=[]):
        self._survey_prop = survey_prop
        self.db_cut = db_cut
        self.db_file = db_file
        self.band_dic = band_dic
        self.add_keys = add_keys
        self.obs_table = self.extract_from_db()

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
        return np.min(self.obs_table['expMJD'])

    @property
    def maxtime(self):
        return np.max(self.obs_table['expMJD'])

    def extract_from_db(self):
        '''Read db file and extract relevant information'''
        dbf = sqlite3.connect(self.db_file)

        keys = ['expMJD',
                'filter',
                'fieldRA',
                'fieldDec',
                'fiveSigmaDepth']+self.add_keys

        where=''
        if self.db_cut != None:
            where=" WHERE "
            for cut_var in self.db_cut:
                where+="("
                for cut in self.db_cut[cut_var]:
                    cut_str=f"{cut}"
                    where+=f"{cut_var}{cut_str} OR "
                where=where[:-4]
                where+=") AND "
            where=where[:-5]
        obs_dic={}
        for k in keys:
            query = 'SELECT '+k+' FROM Summary'+where+';'
            values = dbf.execute(query)
            obs_dic[k] = np.array([a[0] for a in values])
        return Table(obs_dic)

    def epochs_selection(self, SN):
        '''Select epochs that match the survey observations'''
        ModelMinT_obsfrm = SN.sim_model.mintime() * (1 + SN.z)
        ModelMaxT_obsfrm = SN.sim_model.maxtime() * (1 + SN.z)
        ra,dec = SN.coord
        # time selection
        epochs_selec = (self.obs_table['expMJD'] - SN.sim_t0 > ModelMinT_obsfrm) * \
            (self.obs_table['expMJD'] - SN.sim_t0 < ModelMaxT_obsfrm)
        # use to avoid 1e43 errors
        epochs_selec *= (self.obs_table['fiveSigmaDepth'] > 0)
        # Find the index of the field that pass time cut
        epochs_selec_idx = np.where(epochs_selec)
        # Compute the coord of the SN in the rest frame of each field
        ra_size, dec_size = self.field_size
        ra_field_frame, dec_field_frame = change_sph_frame(ra,dec,self.obs_table['fieldRA'][epochs_selec],self.obs_table['fieldDec'][epochs_selec])
        epochs_selec[epochs_selec_idx] *= abs(ra_field_frame) < ra_size/2 # ra selection
        epochs_selec[epochs_selec_idx] *= abs(dec_field_frame) < dec_size/2 # dec selection
        if np.sum(epochs_selec) == 0:
            return None
        return self.make_obs_table(epochs_selec)


    def make_obs_table(self,epochs_selec):
        # Capture noise and filter
        mlim5 = self.obs_table['fiveSigmaDepth'][epochs_selec]
        band = self.obs_table['filter'][epochs_selec].astype('U27')

        # Change band name to correpond with sncosmo bands -> CHANGE EMPLACEMENT
        if self.band_dic is not None:
            band = np.array(list(map(self.band_dic.get,band)))

        if self.zp != 'zp_in_obs':
            zp = [self.zp] * np.sum(epochs_selec)
        elif isinstance(zp,(int, float)):
            zp = self.obs_table['zp'][epochs_selec]
        else:
            raise ValueError("zp is not define")

        # Convert maglim to flux noise (ADU)
        skynoise = pw(10., 0.4 * (self.zp - mlim5)) / 5

        # Create obs table
        obs = Table({'time': self.obs_table['expMJD'][epochs_selec],
                      'band': band,
                      'gain': [self.gain] * np.sum(epochs_selec),
                      'skynoise': skynoise,
                      'zp': zp,
                      'zpsys': ['ab'] * np.sum(epochs_selec)})

        for k in self.add_keys:
            obs[k] = self.obs_table[k][epochs_selec]
        return obs


class SNGen:
    def __init__(self,sim_par, host=None):
        self._sim_par = sim_par
        self.sim_model = self.init_sim_model()
        self.model_keys = ['M0']
        self.model_keys += self.init_model_keys()
        self.host = host

    @property
    def snc_model_par(self):
        return self._sim_par['snc_model_par']

    @property
    def sn_model_par(self):
        return self._sim_par['sn_model_par']

    @property
    def which_model(self):
        info = f"{self.snc_model_par['model_tag']} "
        info+=f"{self.snc_model_par['version']}"
        print(info)
        return

    @property
    def cosmo(self):
        return self._sim_par['cosmo']

    @property
    def cmb(self):
        return self._sim_par['cmb']

    @property
    def snc_model_time(self):
        return self.sim_model.mintime(), self.sim_model.maxtime()

    def init_sim_model(self):
        if self.snc_model_par['model_tag'] == 'salt':
            salt_dir = self.snc_model_par['model_dir']
            if self.snc_model_par['version'] == 2:
                source = snc.SALT2Source(modeldir=salt_dir,name='salt2')
            elif self.snc_model_par['version'] == 3:
                source = snc.SALT3Source(modeldir=salt_dir,name='salt3')
            else :
                raise RuntimeError("Support SALT version = 2 or 3")

        model = snc.Model(source=source)

        if 'smear_mod' in self.snc_model_par:
            smear_mod = self.snc_model_par['smear_mod']
            if smear_mod == 'G10':
                model.add_effect(sct.G10(model),'G10_','rest')

            elif smear_mod[:3] == 'C11':
                if smear_mod == ('C11' or 'C11_0'):
                    model.add_effect(sct.C11(model),'C11_','rest')
                elif smear_mod == 'C11_1':
                    model.add_effect(sct.C11(model),'C11_','rest')
                    model.set(C11_Cuu=1.)
                elif smear_mod == 'C11_2':
                    model.add_effect(sct.C11(model),'C11_','rest')
                    model.set(C11_Cuu=-1.)
        return model

    def init_model_keys(self):
        if self.snc_model_par['model_tag'] == 'salt':
            model_keys = ['alpha','beta']

        return model_keys

    def __call__(self,n_sn,z_range,rand_seed):
        rand_seeds = np.random.default_rng(rand_seed).integers(low=1000, high=100000,size=7)
        t0 = self.gen_peak_time(n_sn,rand_seeds[0])
        mag_smear = self.gen_coh_scatter(n_sn,rand_seeds[4])
        noise_rand_seed = self.gen_noise_rand_seed(n_sn,rand_seeds[5])
        model_par_sncosmo = self.gen_sncosmo_param(n_sn,rand_seeds[6:8])

        if self.host is not None:
            host = self.host.random_host(n_sn,z_range,rand_seeds[1])
            ra = host['ra']
            dec = host['dec']
            zcos = host['redshift']
            vpec = host['vp_sight']
        else:
            ra, dec = self.gen_coord(n_sn,rand_seeds[1])
            zcos = self.gen_zcos(n_sn,z_range,rand_seeds[2])
            vpec = self.gen_vpec(n_sn,rand_seeds[3])

        sn_par = [{'zcos': z,
                   'sim_t0': t,
                   'ra': r,
                   'dec': d,
                   'vpec': v,
                   'mag_smear': ms
                   } for z,t,r,d,v,ms in zip(zcos,t0,ra,dec,vpec,mag_smear)]

        model_default = {}
        for k in self.model_keys:
            model_default[k] = self.sn_model_par[k]

        model_par_list = [{**model_default, 'sncosmo': mpsn, 'noise_rand_seed': rs } for mpsn, rs in zip(model_par_sncosmo, noise_rand_seed)]
        SN_list = [SN(snp,self.cosmo,self.cmb,self.sim_model,mp) for snp,mp in zip(sn_par,model_par_list)]
        return SN_list

    def gen_peak_time(self,n_sn,rand_seed):
        t0 = np.random.default_rng(rand_seed).uniform(*self.sn_model_par['time_range'],size=n_sn)
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
        if self.snc_model_par['model_tag'] == 'salt':
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
        x1_seed,c_seed = np.random.default_rng(rand_seed).integers(low=1000, high=100000,size=2)
        sim_x1 = np.random.default_rng(x1_seed).normal(
            loc=self.sn_model_par['x1_distrib'][0],
            scale=self.sn_model_par['x1_distrib'][1],
            size=n_sn)
        sim_c = np.random.default_rng(c_seed).normal(
            loc=self.sn_model_par['c_distrib'][0], scale=self.sn_model_par['c_distrib'][1], size=n_sn)
        return sim_x1, sim_c

    def gen_vpec(self,n_sn,rand_seed):
        if self.host is None:
            vpec = np.random.default_rng(rand_seed).normal(
                loc=self.sn_model_par['vpec_distrib'][0],
                scale=self.sn_model_par['vpec_distrib'][1],
                size=n_sn)

        else:
            vpec = host.vpec

        return vpec

    def gen_coh_scatter(self,n_sn,rand_seed):
        ''' Generate coherent intrinsic scattering '''
        mag_smear = np.random.default_rng(rand_seed).normal(loc=0, scale=self.sn_model_par['mag_smear'], size=n_sn)
        return mag_smear

    def gen_noise_rand_seed(self,n_sn,rand_seed):
        return np.random.default_rng(rand_seed).integers(low=1000, high=100000,size=n_sn)


class SN:
    '''SN object
    Data structure :
    SN
    ├── _sn_par
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
    ├── epochs
    ├── sim_lc
    └── fit_model
    '''

    def __init__(self,sn_par,cosmo,cmb,sim_model,model_par):
        self.sim_model = sim_model
        self._sn_par = sn_par
        self.cosmo = cosmo
        self.cmb = cmb
        self.model_par = model_par
        self.init_model_par()
        self._epochs = None
        self.sim_lc = None
        self._fit_model = None
        self._ID = None
        return

    @property
    def ID(self):
        if self._ID is None:
            print('No id')
        return self._id

    @ID.setter
    def ID(self,ID):
        if isinstance(ID, int):
            self._ID = ID
        else:
            print('SN ID must mbe an integer')

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
        return self.vpec/c_light_kms

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
    def epochs(self):
         return self._epochs

    @epochs.setter
    def epochs(self,ep_dic):
        self._epochs = ep_dic

    @property
    def z2cmb(self):
        ra_cmb = self.cmb['ra_cmb']
        dec_cmb = self.cmb['dec_cmb']
        v_cmb = self.cmb['v_cmb']

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
        return (1 - v_cmb * (ss + ccc) / c_light_kms) - 1.

    @property
    def sim_mu(self):
        ''' Generate x0/mB parameters for SALT2 '''
        cosmo = FlatLambdaCDM(**self.cosmo)
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

    def pass_cut(self,nep_cut):
        if self.epochs is None:
            return  False
        else:
            for cut in nep_cut:
                cutMin_obsfrm, cutMax_obsfrm = cut[1] * (1 + self.z), cut[2] * (1 + self.z)
                test = (self.epochs['time'] - self.sim_t0 > cutMin_obsfrm)
                test *= (self.epochs['time'] - self.sim_t0 < cutMax_obsfrm)
                if len(cut) == 4:
                    test *= (self.epochs['filter'] == cut[3])
                if np.sum(test) < int(cut[0]):
                    return False
            return True

    def gen_flux(self,add_keys={}):
        ''' Generate simulated flux '''
        params = {**{'z': self.z,'t0': self.sim_t0}, **self.model_par['sncosmo']}
        self.sim_lc = snc.realize_lcs(self.epochs, self.sim_model, [params], scatter=False)[0]
        rs = self.model_par['noise_rand_seed']
        self.sim_lc['flux'] = np.random.default_rng(rs).normal(
                loc=self.sim_lc['flux'], scale=self.sim_lc['fluxerr'])

        for k in add_keys:
            self.sim_lc[k] = obs[k]

        self.reformat_sim_table()


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
            self.sim_lc.meta['sim_mb'] = self.sim_mb
            self.sim_lc.meta['sim_mu'] = self.sim_mu
            self.sim_lc.meta['m_smear'] = self.mag_smear
            if self.ID is not None:
                self.sim_lc.meta['sn_id'] = self.ID
            return

    def get_lc_hdu(self):
        return fits.table_to_hdu(self.sim_lc)
