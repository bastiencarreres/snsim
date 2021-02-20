import sncosmo as snc
import numpy as np
import astropy.units as u
from numpy import power as pw
from astropy.table import Table
from astropy import constants as cst
import yaml
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord

c_light_kms = cst.c.to('km/s').value

class sn_sim :
    def __init__(self,sim_yaml):
        '''Initialisation of the simulation class with the config file'''
        #Default values
        self.dec_cmb = 48.253
        self.ra_cmb = 266.81
        self.v_cmb = 369.82

        with open(sim_yaml, "r") as ymlfile:
           self.sim_cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        #Simulation parameters
        self.data_cfg = self.sim_cfg['data']
        self.obs_cfg_path = self.data_cfg['obs_config_path']

        self.sn_gen = self.sim_cfg['sn_gen']
        self.n_sn =  int(self.sn_gen['n_sn'])

        if 'v_cmb' in self.sn_gen:
            self.v_cmb = self.sn_gen['v_cmb']

        #Cosmology parameters
        self.cosmo_cfg = self.sim_cfg['cosmology']
        self.cosmo = FlatLambdaCDM(H0=self.cosmo_cfg['H0'], Om0=self.cosmo_cfg['Om'])

        #Salt2 parameters
        self.salt2_gen = self.sim_cfg['salt2_gen']

        #Vpec parameters
        self.vpec_gen = self.sim_cfg['vpec_gen']

    def open_obs(self):
        return

    def simulate(self):
        '''Simulation routine'''
        self.gen_param_array()
        return

    def gen_param_array(self):
        #Init randseed in order to reproduce SNs
        self.randseed = int(self.sn_gen['randseed'])
        randseeds = np.random.default_rng(self.randseed).integers(low=1000,high=10000,size=6)
        self.randseeds = {'z_seed': randseeds[0],
                          'x0_seed': randseeds[1],
                          'x1_seed': randseeds[2],
                          'c_seed': randseeds[3],
                          'coord_seed': randseeds[4],
                          'vpec_seed': randseeds[5]
                          }
        #Init z range
        self.z_range = self.sn_gen['z_range']

        #Init vpec_gen
        self.mean_vpec = self.vpec_gen['mean_vpec']
        self.sig_vpec = self.vpec_gen['sig_vpec']

        #Init M0
        self.M0 = self.sn_gen['M0']

        #Init x1 and c
        self.mean_x1=self.salt2_gen['mean_x1']
        self.sig_x1=self.salt2_gen['sig_x1']

        self.mean_c = self.salt2_gen['mean_c']
        self.sig_c = self.salt2_gen['sig_c']

        #Redshift generation
        self.gen_redshift_cos()
        self.gen_coord()
        self.gen_z2cmb()
        self.gen_z_pec()
        self.zobs = (1+self.zcos)*(1+self.zpec)*(1+self.z2cmb)

        #SALT2 params generation
        self.gen_sn_par()
        self.gen_sn_mag()

        #self.sim_t0=np.zeros(self.n_sn)
        self.sim_t0=np.array([56188]*self.n_sn)
        self.params = [{'z': z,
                  't0': peak,
                  'x0': x0,
                  'x1': x1,
                  'c': c
                  } for z,peak,x0,x1,c in zip(self.zobs,self.sim_t0,self.sim_x0,self.sim_x1,self.sim_c)]

    def gen_redshift_cos(self):
        self.zcos = np.random.default_rng(self.randseeds['z_seed']).uniform(low=self.z_range[0],high=self.z_range[1],size=self.n_sn)
        return

    def gen_coord(self):
        # extract ra dec from obs config
        seeds = np.random.default_rng(self.randseeds['coord_seed']).integers(low=1000,high=10000,size=2)
        self.randseeds['ra_seed'] = seeds[0]
        self.randseeds['dec_seed']=seeds[1]
        self.ra = np.random.default_rng(self.randseeds['ra_seed']).uniform(low=0,high=2*np.pi,size=self.n_sn)
        self.dec = np.random.default_rng(self.randseeds['dec_seed']).uniform(low=-np.pi/2,high=np.pi/2,size=self.n_sn)
        return

    def gen_z2cmb(self):
        # use ra dec to simulate the effect of our motion
        coordfk5 = SkyCoord(self.ra*u.rad, self.dec*u.rad, frame='fk5') #coord in fk5 frame
        galac_coord = coordfk5.transform_to('galactic')
        self.ra_gal=galac_coord.l.rad-2*np.pi*np.sign(galac_coord.l.rad)*(abs(galac_coord.l.rad)>np.pi)
        self.dec_gal=galac_coord.b.rad

        ss = np.sin(self.dec_gal)*np.sin(self.dec_cmb*np.pi/180)
        ccc = np.cos(self.dec_gal)*np.cos(self.dec_cmb*np.pi/180)*np.cos(self.ra_gal-self.ra_cmb*np.pi/180)
        self.z2cmb = (1+self.zcos)*(1-self.v_cmb*(ss+ccc)/c_light_kms)-1.
        return

    def gen_z_pec(self):
        self.vpec = np.random.default_rng(self.randseeds['vpec_seed']).normal(loc=self.mean_vpec,scale=self.sig_vpec,size=self.n_sn)
        self.zpec = self.vpec/c_light_kms
        return

    def gen_sn_par(self):
        ''' Generate x1 and c for the SALT2 model'''
        self.sim_x1 = np.random.default_rng(self.randseeds['x1_seed']).normal(loc=self.mean_x1,scale=self.sig_x1,size=self.n_sn)
        self.sim_c = np.random.default_rng(self.randseeds['c_seed']).normal(loc=self.mean_c,scale=self.sig_c,size=self.n_sn)
        return

    def gen_sn_mag(self):
        ''' Generate x0/mB parameters for SALT2 '''
        self.mag_smear = 0
        self.sim_mu = 5*np.log10((1+self.zcos)*(1+self.z2cmb)*pw((1+self.zpec),2)*self.cosmo.comoving_distance(self.zcos).value)+25
        self.sim_mB = self.sim_mu + self.M0 + self.mag_smear
        self.sim_x0 = pw(10,-0.4*(self.sim_mB-10.5020699)) #10.5020699 is an offset
        return

    def gen_flux(self):
        obs = Table({'time': [56176.19, 56188.254, 56207.172],
             'band': ['desg', 'desr', 'desi'],
             'gain': [1., 1., 1.],
             'skynoise': [191.27, 147.62, 160.40],
             'zp': [30., 30., 30.],
             'zpsys':['ab', 'ab', 'ab']})
        model=snc.Model(source='salt2-extended')
        return snc.realize_lcs(obs, model, self.params)
