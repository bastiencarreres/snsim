import sncosmo as snc
import numpy as np
from numpy import power as pw
from astropy.table import Table
from astropy import constants as cst
import yaml
from astropy.cosmology import FlatLambdaCDM

c_light_kms = cst.c.to('km/s').value

class sn_sim :
    def __init__(self,sim_yaml):
        '''Initialisation of the simulation class with the config file'''
        with open(sim_yaml, "r") as ymlfile:
           self.sim_cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        #Simulation parameters
        self.data_cfg = self.sim_cfg['data']
        self.obs_cfg_path = self.data_cfg['obs_config_path']

        self.sn_gen = self.sim_cfg['sn_gen']
        self.n_sn =  int(self.sn_gen['n_sn'])

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

        #Init randseed
        self.randseed = int(self.sn_gen['randseed'])
        randseeds = np.random.default_rng(self.randseed).integers(low=1000,size=6)
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

        self.t0= 0
        params = {'z': self.zobs,
                  't0': self.t0,
                  'x0': self.x0,
                  'x1': self.sim_x1,
                  'c': self.sim_c
                  }

    def gen_redshift_cos(self):
        self.zcos = np.random.default_rng(self.randseeds['z_seed']).uniform(low=self.z_range[0],high=self.z_range[1],size=self.n_sn)
        return

    def gen_coord(self):
        # extract ra dec from obs config
        return

    def gen_z2cmb(self):
        # use ra dec to simulate the effect of our motion*
        self.z2cmb = 0
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
        ''' Generate x0 parameter for SALT2 '''
        self.mag_smear = 0
        self.mu_sim = 5*np.log10((1+self.zcos)*(1+self.z2cmb)*pw((1+self.zpec),2)*self.cosmo.comoving_distance(self.zcos).value)+25
        self.mB = self.mu_sim + self.M0 + self.mag_smear
        self.x0 = pw(10,-0.4*(self.mB-10.5020699)) #10.502069945029266 is an offset
        return


    def gen_flux(obs):
        return
