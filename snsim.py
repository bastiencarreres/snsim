import sncosmo as snc
import numpy as np
import astropy.units as u
from numpy import power as pw
from astropy.table import Table
from astropy import constants as cst
from astropy.io import fits
import yaml
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import time

c_light_kms = cst.c.to('km/s').value
snc_mag_offset = 10.5020699 #just an offset -> set_peakmag(mb=0,'bessellb', 'ab') -> offset=2.5*log10(get_x0) change with magsys

def x0_to_mB(par,inv): #Faire 2 fonctions
    if inv == 0:
        return -2.5*np.log10(par)+snc_mag_offset
    else:
        return pw(10,-0.4*(par-snc_mag_offset))

def box_output(sep,line):
    l = len(sep)-len(line)-2
    space1 = ' '*(l//2)
    space2 = ' '*(l//2+l%2)
    return '#'+space1+line+space2+'#'

def snc_fit(lc,model):
    return snc.fit_lc(lc, model, ['t0', 'x0', 'x1', 'c'])

def plot_lc(flux_table,bands,zp=25.,mag=False,model=None):

    flux_norm, fluxerr_norm = norm_flux(flux_table,zp)
    time = flux_table['time']

    t0= flux_table.meta['t0']
    z = flux_table.meta['z']
    if salt2_par is None:
        x0 = flux_table.meta['x0']
        mb = x0_to_mB(flux_table.meta['x0'],0)
        x1 = flux_table.meta['x1']
        c = flux_table.meta['c']
    else :

    if model is not None:
        time_th = np.linspace(t0-20, t0+30,100)
        model.set(z=z, c=c, t0=t0, x0=x0, x1=x1)

    title = f'$m_B$ = {mb:.3f} $x_1$ = {x1:.3f} $c$ = {c:.4f}'

    plt.figure()
    plt.title(title)
    plt.xlabel('Time to peak')


    for b in bands:
        band_mask = flux_table['band']==b
        flux_b = flux_norm[band_mask]
        fluxerr_b = fluxerr_norm[band_mask]
        time_b = time[band_mask]
        if mag:
            plt.gca().invert_yaxis()
            plt.ylabel('Mag')
            flux_b = flux_b[flux_b>0] #Delete < 0 pts
            time_b=time_b[flux_b>0]
            plot = -2.5*np.log10(flux_b)+zp
            err = 2.5/np.log(10)*1/flux_b*fluxerr_b
            if model is not None:
                plot_th = model.bandmag(b,'ab',time_th)
        else:
            plt.ylabel('Flux')
            plot = flux_b
            err = fluxerr_b
            if model is not None:
                plot_th=model.bandflux(b,time_th,zp=zp,zpsys='ab')
        p = plt.errorbar(time_b-t0,plot,yerr=err,label=b,fmt='o')
        if model is not None:
            plt.plot(time_th-t0,plot_th, color=p[0].get_color())
    plt.legend()
    plt.show()
    return

def norm_flux(flux_table,zp):
    '''Taken from sncosmo -> set the flux to the same zero-point'''
    norm_factor = pw(10,0.4*(zp-flux_table['zp']))
    flux_norm=flux_table['flux']*norm_factor
    fluxerr_norm = flux_table['fluxerr']*norm_factor
    return flux_norm,fluxerr_norm

class sn_sim :
    def __init__(self,sim_yaml):
        '''Initialisation of the simulation class with the config file'''
        #Default values
        self.yml_path = sim_yaml
        #CMB values
        self.dec_cmb = 48.253
        self.ra_cmb = 266.81
        self.v_cmb = 369.82

        with open(sim_yaml, "r") as ymlfile:
           self.sim_cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        #Simulation parameters
        self.data_cfg = self.sim_cfg['data']
        self.obs_cfg_path = self.data_cfg['obs_config_path']
        self.write_path = self.data_cfg['write_path']
        self.sim_name = self.data_cfg['sim_name']

        self.sn_gen = self.sim_cfg['sn_gen']
        self.n_sn = int(self.sn_gen['n_sn'])

        if 'v_cmb' in self.sn_gen:
            self.v_cmb = self.sn_gen['v_cmb']

        #Cosmology parameters
        self.cosmo_cfg = self.sim_cfg['cosmology']
        self.cosmo = FlatLambdaCDM(H0=self.cosmo_cfg['H0'], Om0=self.cosmo_cfg['Om'])

        #Salt2 parameters
        self.salt2_gen = self.sim_cfg['salt2_gen']
        self.alpha = self.salt2_gen['alpha']
        self.beta = self.salt2_gen['beta']
        self.salt2_dir = self.salt2_gen['salt2_dir']

        source = snc.SALT2Source(modeldir=self.salt2_dir)
        self.model=snc.Model(source=source)
        #Vpec parameters
        self.vpec_gen = self.sim_cfg['vpec_gen']

        self.open_obs_header()


    def simulate(self):
        '''Simulation routine'''

        print('-----------------------------------')
        print(f'SIM NAME : {self.sim_name}')
        print(f'CONFIG FILE : {self.yml_path}')
        print(f'OBS FILE : {self.obs_cfg_path}')
        print(f'SIM WRITE DIRECTORY : {self.write_path}')
        print(f'-----------------------------------\n')
        start_time = time.time()

        self.obs=[]
        self.obs_header=[]
        with fits.open(self.obs_cfg_path) as hduf:
            for hdu in hduf[1:]:
                self.obs.append(hdu.data)
                self.obs_header.append(hdu.header)
        sep='###############################################'
        sep2 =box_output(sep,'------------')
        line = f'OBS FILE read in {time.time()-start_time:.1f} seconds'
        print(sep)
        print(box_output(sep,line))
        print(sep2)
        sim_time = time.time()
        #Generate z, x0, x1, c
        self.gen_param_array()
        #Simulate for each obs
        self.gen_flux()

        l=f'{self.n_sn} SN lcs generated in {time.time() - sim_time:.1f} seconds'
        print(box_output(sep,l))
        print(sep2)

        write_time = time.time()
        self.write_sim()
        l=f'Sim file write in {time.time() - write_time:.1f} seconds'
        print(box_output(sep,l))
        print(sep2)
        l=f'SIMULATION TERMINATED in {time.time() - start_time:.1f} seconds'
        print(box_output(sep,l))
        print(sep)
        return

    def gen_param_array(self):
        #Init randseed in order to reproduce SNs
        if 'randseed' in self.sn_gen:
            self.randseed = int(self.sn_gen['randseed'])
        else:
            self.randseed = np.random.randint(low=1000,high=10000)

        randseeds = np.random.default_rng(self.randseed).integers(low=1000,high=10000,size=7)
        self.randseeds = {'z_seed': randseeds[0],
                          'x0_seed': randseeds[1],
                          'x1_seed': randseeds[2],
                          'c_seed': randseeds[3],
                          'coord_seed': randseeds[4],
                          'vpec_seed': randseeds[5],
                          'smearM_seed': randseeds[6]
                          }
        #Init z range
        self.z_range = self.sn_gen['z_range']
        self.sigmaM = self.sn_gen['mag_smear'] # To change

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
        self.zCMB = (1+self.zcos)*(1+self.zpec)-1.
        self.zobs = (1+self.zcos)*(1+self.zpec)*(1+self.z2cmb)-1.

        #SALT2 params generation
        self.gen_sn_par()
        self.gen_sn_mag()

        #self.sim_t0=np.zeros(self.n_sn)
        #Total fake for the moment....
        self.sim_t0=np.array([52000+20+30*i for i in range(self.n_sn)])
        self.params = [{'z': z,
                  't0': peak,
                  'x0': x0,
                  'x1': x1,
                  'c': c
                  } for z,peak,x0,x1,c in zip(self.zobs,self.sim_t0,self.sim_x0,self.sim_x1,self.sim_c)]


    def open_obs_header(self):
        ''' Open the fits obs file header'''
        with fits.open(self.obs_cfg_path,'readonly') as obs_fits:
            self.obs_header_main = obs_fits[0].header
            self.bands = self.obs_header_main['bands'].split()
        return

    def gen_redshift_cos(self):
        '''Function to get zcos, to be updated'''
        self.zcos = np.random.default_rng(self.randseeds['z_seed']).uniform(low=self.z_range[0],high=self.z_range[1],size=self.n_sn)
        return

    def gen_coord(self):
        # extract ra dec from obs config
        self.ra = []
        self.dec = []
        for i in range(self.n_sn):
            obs=self.obs_header[i]
            self.ra.append(obs['RA'])
            self.dec.append(obs['DEC'])

        #seeds = np.random.default_rng(self.randseeds['coord_seed']).integers(low=1000,high=10000,size=2)
        #self.randseeds['ra_seed'] = seeds[0]
        #self.randseeds['dec_seed']=seeds[1]
        #self.ra = np.random.default_rng(self.randseeds['ra_seed']).uniform(low=0,high=2*np.pi,size=self.n_sn)
        #self.dec = np.random.default_rng(self.randseeds['dec_seed']).uniform(low=-np.pi/2,high=np.pi/2,size=self.n_sn)
        return

    def gen_z2cmb(self):
        # use ra dec to simulate the effect of our motion
        coordfk5 = SkyCoord(self.ra*u.deg, self.dec*u.deg, frame='fk5') #coord in fk5 frame
        galac_coord = coordfk5.transform_to('galactic')
        self.ra_gal=galac_coord.l.rad-2*np.pi*np.sign(galac_coord.l.rad)*(abs(galac_coord.l.rad)>np.pi)
        self.dec_gal=galac_coord.b.rad

        ss = np.sin(self.dec_gal)*np.sin(self.dec_cmb*np.pi/180)
        ccc = np.cos(self.dec_gal)*np.cos(self.dec_cmb*np.pi/180)*np.cos(self.ra_gal-self.ra_cmb*np.pi/180)
        self.z2cmb = (1-self.v_cmb*(ss+ccc)/c_light_kms)-1.
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
        self.mag_smear =  np.random.default_rng(self.randseeds['smearM_seed']).normal(loc=0,scale=self.sigmaM,size=self.n_sn)
        self.sim_mu = 5*np.log10((1+self.zcos)*(1+self.z2cmb)*pw((1+self.zpec),2)*self.cosmo.comoving_distance(self.zcos).value)+25
        #Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 + beta*c : scattering due to color and stretch} + {intrinsic smearing}
        self.sim_mB = self.sim_mu + self.M0 - self.alpha*self.sim_x1 + self.beta*self.sim_c + self.mag_smear
        self.sim_x0 = x0_to_mB(self.sim_mB,1)
        return

    def gen_flux(self):
        ''' Generate simulated flux '''
        self.sim_lc=[snc.realize_lcs(obs, self.model, [params],scatter=False)[0] for obs,params in zip(self.obs,self.params)]
        return

    def plot_simlc(self,lc_id,zp=25.,mag=False):
        plot_lc(self.sim_lc[lc_id],['ztfg','ztfr','ztfi'],zp=zp,mag=mag,model=self.model)
        return

    def fit_lcs(self):
        self.fit_res = []
        for i in range(self.n_sn):
            self.model.set(z=self.sim_lc[i].meta['z'])  # set the model's redshift.
            self.fit_res.append(snc_fit(self.sim_lc[i],self.model))
        return

    def write_sim(self):
        '''Write the simulated lc in a fits file'''
        lc_hdu_list = []
        for i,tab in enumerate(self.sim_lc):
            tab.meta['vpec'] = self.vpec[i]
            tab.meta['zcos'] = self.zcos[i]
            tab.meta['zpec'] = self.zpec[i]
            tab.meta['z2cmb'] = self.z2cmb[i]
            tab.meta['zCMB'] = self.zCMB[i]
            tab.meta['ra'] = self.ra[i]
            tab.meta['dec'] = self.dec[i]
            lc_hdu_list.append(fits.table_to_hdu(tab))

        hdu_list = fits.HDUList([fits.PrimaryHDU(header=fits.Header({'n_obs': self.n_sn}))]+lc_hdu_list)
        hdu_list.writeto(self.write_path+self.sim_name+'.fits',overwrite=True)
        return

class open_sim:
    def __init__(self,sim_file,SALT2_dir):
        self.salt2_dir = SALT2_dir
        source = snc.SALT2Source(modeldir=self.salt2_dir)
        self.model=snc.Model(source=source)

        self.lc=[]

        with fits.open(sim_file) as sf:
            self.n_sn=sf[0].header['N_OBS']
            for hdu in sf[1:]:
                data=hdu.data
                tab= Table(data)
                tab.meta=hdu.header
                self.lc.append(tab)
        return

    def fit_lc(self):
        self.fit_res=[]
        for i in range(self.n_sn):
            self.model.set(z=self.lc[i].meta['z'])  # set the model's redshift.
            self.fit_res.append(snc_fit(self.lc[i],self.model))
        return
