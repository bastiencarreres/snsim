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
import sqlite3
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sn_sim_print = '     _______..__   __.         _______. __  .___  ___. \n'
sn_sim_print+= '    /       ||  \ |  |        /       ||  | |   \/   | \n'
sn_sim_print+= '   |   (----`|   \|  |       |   (----`|  | |  \  /  | \n'
sn_sim_print+= '    \   \    |  . `  |        \   \    |  | |  |\/|  | \n'
sn_sim_print+= '.----)   |   |  |\   |    .----)   |   |  | |  |  |  | \n'
sn_sim_print+= '|_______/    |__| \__|    |_______/    |__| |__|  |__| \n'

c_light_kms = cst.c.to('km/s').value
snc_mag_offset = 10.5020699 #just an offset -> set_peakmag(mb=0,'bessellb', 'ab') -> offset=2.5*log10(get_x0) change with magsys

def x0_to_mB(x0):
    '''Convert x0 to mB'''
    return -2.5*np.log10(x0)+snc_mag_offset

def mB_to_x0(mB):
    '''Convert mB to x0'''
    return pw(10,-0.4*(mB-snc_mag_offset))

def box_output(sep,line):
    '''Use for plotting simulation output'''
    l = len(sep)-len(line)-2
    space1 = ' '*(l//2)
    space2 = ' '*(l//2+l%2)
    return '#'+space1+line+space2+'#'

def snc_fit(lc,model):
    '''Fit the given lc with the given SALT2 model
       Free parameters are : - The SN peak magnitude in B-band t0
                             - The normalisation factor x0 (<=> mB)
                             - The stretch parameter x1
                             - The color parameter c
    '''
    return snc.fit_lc(lc, model, ['t0', 'x0', 'x1', 'c'], modelcov=True)

def compute_fit_error(fit_model,cov,band,flux_th,time_th,zp,magsys='ab'):
    '''Compute theorical fluxerr from fit err = sqrt(COV)
    where COV = J**T * COV(x0,x1,c) * J with J = (dF/dx0, dF/dx1, dF/dc) the jacobian.
    According to Fnorm = x0/(1+z) * int_\lambda (M0(lambda_s,p)+x1*M1(lambda_s,p))*10**(-0.4*c*CL(lambda_s)) * T_b(lambda) * lambda/hc dlambda * norm_factor
    where norm_factor = 10**(0.4*ZP_norm)/ZP_magsys. We found :
    dF/dx0 = F/x0
    dF/dx1 = x0/(1+z) * int_lambda M1(lambda_s,p))*10**(-0.4*c*CL(lambda_s)) * T_b(lambda) * lambda/hc dlambda * norm_factor
    dF/dc  =  -0.4*ln(10)*x0/(1+z) * int_\lambda (M0(lambda_s,p)+x1*M1(lambda_s,p))*CL(lambda_s)*10**(-0.4*c*CL(lambda_s)) * T_b(lambda) * lambda/hc dlambda * norm_factor
    '''
    a = 1./(1+fit_model.parameters[0])
    t0 = fit_model.parameters[1]
    x0 = fit_model.parameters[2]
    x1 = fit_model.parameters[3]
    c = fit_model.parameters[4]
    COV  = cov
    b = snc.get_bandpass(band)
    wave, dwave = snc.utils.integration_grid(b.minwave(), b.maxwave(),snc.constants.MODEL_BANDFLUX_SPACING)
    trans = b(wave)
    ms =  snc.get_magsystem(magsys)
    zpms = ms.zpbandflux(b)
    normfactor = 10**(0.4*zp)/zpms
    err_th = []
    for t,f in zip(time_th,flux_th):
        p = time_th-t0
        dfdx0 = f/x0
        fint1 = fit_model.source._model['M1'](a*p, a*wave)[0]*10.**(-0.4*fit_model.source._colorlaw(a*wave)*c)
        fint2 = (fit_model.source._model['M0'](a*p, a*wave)[0]+x1*fit_model.source._model['M1'](a*p, a*wave)[0])*10.**(-0.4*fit_model.source._colorlaw(a*wave)*c)*fit_model.source._colorlaw(a*wave)
        m1int = np.sum(wave * trans * fint1, axis=0) * dwave / snc.constants.HC_ERG_AA
        clint = np.sum(wave*trans*fint2, axis=0) * dwave / snc.constants.HC_ERG_AA
        dfdx1 = a*x0*m1int*normfactor
        dfdc = -0.4*np.log(10)*a*x0*clint*normfactor
        J = np.asarray([dfdx0,dfdx1,dfdc],dtype=float)
        err = np.sqrt(J.T @ cov @ J)
        err_th.append(err)
    err_th = np.asarray(err_th)
    return err_th

def plot_lc(flux_table,zp=25.,mag=False,sim_model=None,fit_model=None,fit_cov=None,residuals=False):
    '''General plot function
       Options : - zp = float, use the normalisation zero point that you want (default: 25.)
                 - mag = boolean, plot magnitude (default = False)
    '''

    bands = find_filters(flux_table['band'])
    flux_norm, fluxerr_norm = norm_flux(flux_table,zp)
    time = flux_table['time']

    t0= flux_table.meta['t0']
    z = flux_table.meta['z']

    time_th = np.linspace(t0-19.8*(1+z),t0+49.8*(1+z),500)

    fig=plt.figure()
    if residuals:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1],sharex=ax0)
    else :
        ax0 = plt.subplot(111)

    if sim_model is not None:
        x0 = flux_table.meta['x0']
        mb = x0_to_mB(flux_table.meta['x0'])
        x1 = flux_table.meta['x1']
        c = flux_table.meta['c']

        sim_model.set(z=z, c=c, t0=t0, x0=x0, x1=x1)


        title = f'z = {z:.3f} $m_B$ = {mb:.3f} $x_1$ = {x1:.3f} $c$ = {c:.4f}'
        ax0.set_title(title)



    plt.xlabel('Time to peak')
    ylim = 0
    for b in bands:
        band_mask = flux_table['band']==b
        flux_b = flux_norm[band_mask]
        fluxerr_b = fluxerr_norm[band_mask]
        time_b = time[band_mask]

        if mag:
            plt.gca().invert_yaxis()
            ax0.ylabel('Mag')
            flux_b, fluxerr_b, time_b = flux_b[flux_b>0], fluxerr_b[flux_b>0], time_b[flux_b>0] #Delete < 0 pts
            plot = -2.5*np.log10(flux_b)+zp
            plt.ylim(np.max(plot)+3,np.min(plot)-3)
            err = 2.5/np.log(10)*1/flux_b*fluxerr_b
            if sim_model is not None:
                plot_th = sim_model.bandmag(b,'ab',time_th)
            if fit_model is not None:
                plot_fit = fit_model.bandmag(b,'ab',time_th)
                if fit_cov is not None:
                    err_th = compute_fit_error(fit_model,fit_cov,b,plot_fit,time_th,zp)
                    err_th = 2.5/(np.log(10)*pw(10,-0.4*(plot_fit-zp)))*err_th
                if residuals:
                    fit_pts = fit_model.bandmag(b,'ab',time_b)
                    rsd = plot-fit_pts

        else:
            ax0.set_ylabel('Flux')
            plot = flux_b
            err = fluxerr_b
            if sim_model is not None:
                plot_th = sim_model.bandflux(b,time_th,zp=zp,zpsys='ab')
                ylim = ylim+(np.max(plot_th)-ylim)*(np.max(plot_th)>ylim)
            if fit_model is not None:
                plot_fit = fit_model.bandflux(b,time_th,zp=zp,zpsys='ab')
                if fit_cov is not None:
                    err_th = compute_fit_error(fit_model,fit_cov,b,plot_fit,time_th,zp)
                if residuals:
                    fit_pts = fit_model.bandflux(b,time_b,zp=zp,zpsys='ab')
                    rsd = plot-fit_pts

        p = ax0.errorbar(time_b-t0,plot,yerr=err,label=b,fmt='o',markersize=2.5)
        if sim_model is not None:
            ax0.plot(time_th-t0,plot_th, color=p[0].get_color())
        if fit_model is not None:
            ax0.plot(time_th-t0, plot_fit,color=p[0].get_color(),ls='--')
            if fit_cov is not None:
                ax0.fill_between(time_th-t0, plot_fit-err_th, plot_fit+err_th,alpha=0.5)
            if residuals :
                ax1.set_ylabel('Data - Model')
                ax1.errorbar(time_b-t0,rsd,yerr=err,fmt='o')
                ax1.axhline(0,ls='dashdot',c='black',lw=1.5)
                ax1.set_ylim(-np.max(abs(rsd))*2,np.max(abs(rsd))*2)
                ax1.plot(time_th-t0,err_th,ls='--',color=p[0].get_color())
                ax1.plot(time_th-t0,-err_th,ls='--',color=p[0].get_color())

    #plt.ylim(-np.max(ylim)*0.1,np.max(ylim)*1.1)
    handles, labels = ax0.get_legend_handles_labels()
    if sim_model is not None:
        sim_line = Line2D([0], [0], color='k', linestyle='solid')
        sim_label = 'Sim'
        handles.append(sim_line)
        labels.append(sim_label)

    if fit_model is not None:
        fit_line = Line2D([0], [0], color='k', linestyle='--')
        fit_label = 'Fit'
        handles.append(fit_line)
        labels.append(fit_label)

    ax0.axhline(ls='dashdot',c='black',lw=1.5)
    ax0.legend(handles=handles,labels=labels)
    plt.subplots_adjust(hspace=.0)
    plt.show()
    return

def find_filters(filter_table):
    '''Take a list of obs filter and return the name of the different filters'''
    filter_list = []
    for f in filter_table:
        if f not in filter_list:
            filter_list.append(f)
    return filter_list

def norm_flux(flux_table,zp):
    '''Taken from sncosmo -> set the flux to the same zero-point'''
    norm_factor = pw(10,0.4*(zp-flux_table['zp']))
    flux_norm = flux_table['flux']*norm_factor
    fluxerr_norm = flux_table['fluxerr']*norm_factor
    return flux_norm,fluxerr_norm

def add_filter(path):#Not implemented yet for later purpose
    input_name={}
    for band in bands:
        table = np.loadtxt(band[1])
        name= band[0]
        band = snc.Bandpass(wavelength, transmission, name=name)
        try:
            snc.register(band)
        except (Exception):
            band.name += '_temp'
            snc.register(band,force=True)
            input_name[band[0]] = band.name
    if input_name == {}:
        return None
    else:
        return input_name

class sn_sim :
    def __init__(self,sim_yaml):
        '''Initialisation of the simulation class with the config file
        config.yml

        NOTE : - obs_file and db_file are optional but you must set one of the two!!!
               - If the name of bands in the obs/db file doesn't match sncosmo bands
            you can use the key band_dic to translate filters names
               - If you don't set the filter name item in nep_cut, the cut apply to all the bands

        +----------------------------------------------------------------------------------+
        | data :                                                                           |
        |     write_path: '/PATH/TO/OUTPUT'                                                |
        |     sim_name: 'NAME OF SIMULATION'                                               |
        |     band_dic: {'r':'ztfr','g':'ztfg','i':'ztfi'} #(Optional -> if bandname in    |
        | db/obs file doesn't correpond to those in sncosmo registery)                     |
        |     obs_config_path: '/PATH/TO/OBS/FILE' #(Optional -> use db_file)              |
        | db_config: #(Optional -> use obs_file)                                           |
        |     dbfile_path: '/PATH/TO/FILE'                                                 |
        |     zp: INSTRUMENTAL ZEROPOINT                                                   |
        |     gain: CCD GAIN e-/ADU                                                        |
        | sn_gen:                                                                          |
        |     n_sn: NUMBER OF SN TO GENERATE                                               |
        |     nep_cut: [[nep_min1,Tmin,Tmax],[nep_min2,Tmin2,Tmax2,'filter1'],...] EP CUTS |
        |     randseed: RANDSEED TO REPRODUCE SIMULATION #(Optional)                       |
        |     z_range: [ZMIN,ZMAX]                                                         |
        |     v_cmb: OUR PECULIAR VELOCITY #(Optional, default = 369.82 km/s)              |
        |     M0: SN ABSOLUT MAGNITUDE                                                     |
        |     mag_smear: SN INTRINSIC SMEARING                                             |
        | cosmology:                                                                       |
        |     Om: MATTER DENSITY                                                           |
        |     H0: HUBBLE CONSTANT                                                          |
        | salt2_gen:                                                                       |
        |     salt2_dir: '/PATH/TO/SALT2/MODEL'                                            |
        |     alpha: STRETCH CORRECTION = alpha*x1                                         |
        |     beta: COLOR CORRECTION = -beta*c                                             |
        |     mean_x1: MEAN X1 VALUE                                                       |
        |     mean_c: MEAN C VALUE                                                         |
        |     sig_x1: SIGMA X1                                                             |
        |     sig_c: SIGMA C                                                               |
        | vpec_gen:                                                                        |
        |     mean_vpec: MEAN SN PECULIAR VEL                                              |
        |     sig_vpec: SIGMA VPEC                                                         |
        |                                                                                  |
        +----------------------------------------------------------------------------------+
        '''
    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #----------------- DEFAULT VALUES -----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        #Default values
        self.yml_path = sim_yaml
        #CMB values
        self.dec_cmb = 48.253
        self.ra_cmb = 266.81
        self.v_cmb = 369.82


        with open(sim_yaml, "r") as ymlfile:
           self.sim_cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #----------- data and db_config section -----------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        #Simulation parameters
        self.data_cfg = self.sim_cfg['data']

        #Condition to use obs_file or db_file
        if 'db_config' in self.sim_cfg and 'obs_config_path' in self.data_cfg:
            raise RuntimeError("The simulation can't run with obs file and db file, just set one of the two")
        elif 'obs_config_path' in self.data_cfg:
            self.use_obs = True
        else:
            if 'db_config' in self.sim_cfg:
                self.use_obs = False
            else:
                raise RuntimeError("Set a db_file or a obs_file -> type help(sn_sim) to print the syntax")

        #Initialisation of db/obs_path
        if self.use_obs:
            self.obs_cfg_path = self.data_cfg['obs_config_path']
            self.open_obs_header()
        else:
            self.db_cfg = self.sim_cfg['db_config']
            self.db_file= self.db_cfg['dbfile_path']
            self.zp = self.db_cfg['zp']
            self.gain= self.db_cfg['gain']

        self.write_path = self.data_cfg['write_path']
        self.sim_name = self.data_cfg['sim_name']

        #Band dic : band_name_obs/db_file -> band_name_sncosmo
        if 'band_dic' in self.data_cfg:
            self.band_dic = self.data_cfg['band_dic']
        else:
            self.band_dic = None

    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #----------------- sn_gen section -----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        self.sn_gen = self.sim_cfg['sn_gen']
        self.n_sn = int(self.sn_gen['n_sn'])


    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #--------------- cosmomogy section ----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        #Cosmology parameters
        self.cosmo_cfg = self.sim_cfg['cosmology']
        self.cosmo = FlatLambdaCDM(H0=self.cosmo_cfg['H0'], Om0=self.cosmo_cfg['Om'])


    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #--------------- salt2_gen section ----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        #Salt2 parameters
        self.salt2_gen = self.sim_cfg['salt2_gen']
        self.alpha = self.salt2_gen['alpha']
        self.beta = self.salt2_gen['beta']
        self.salt2_dir = self.salt2_gen['salt2_dir']

        source = snc.SALT2Source(modeldir=self.salt2_dir)
        self.model=snc.Model(source=source)

    #++++++++++++++++++++++++++++++++++++++++++++++++++#
    #--------------- vpec_gen section -----------------#
    #++++++++++++++++++++++++++++++++++++++++++++++++++#

        self.vpec_gen = self.sim_cfg['vpec_gen']


    #Init fit_res_table
        self.fit_res = np.asarray(['No_fit']*self.n_sn,dtype='object')

    #Minimal nbr of epochs in LC
        if 'nep_cut' in self.sn_gen:
            if isinstance(self.sn_gen['nep_cut'], (int,float)):
                self.nep_cut = [(self.sn_gen['nep_cut'],self.model.mintime(),self.model.maxtime())]
                print(self.nep_cut)
            elif isinstance(self.sn_gen['nep_cut'], (list)):
                self.nep_cut = self.sn_gen['nep_cut']

        else:
            self.nep_cut = [(1,self.model.mintime(),self.model.maxtime())]

        if 'v_cmb' in self.sn_gen:
            self.v_cmb = self.sn_gen['v_cmb']
        return

    def simulate(self):
        '''Simulation routine :
        1- READ OBS/DB FILE
        2- GEN REDSHIFT AND SALT2 PARAM
        3- GEN LC FLUX WITH sncosmo
        4- WRITE LC TO A FITS FILE
        '''
        print(sn_sim_print)
        print('-----------------------------------')
        print(f'SIM NAME : {self.sim_name}')
        print(f'CONFIG FILE : {self.yml_path}')

        if self.use_obs:
            print(f'OBS FILE : {self.obs_cfg_path}')
        else:
            print(f'DB FILE : {self.db_file}')

        print(f'SIM WRITE DIRECTORY : {self.write_path}')
        print(f'-----------------------------------\n')
        start_time = time.time()

        self.obs=[]
        if self.use_obs:
            self.obs_header=[]
            with fits.open(self.obs_cfg_path) as hduf:
                for hdu in hduf[1:]:
                    if self.band_dic is not None:
                        for i,b in hdu['band']:
                            hdu.data['band'][i] = self.band_dic[b]
                    self.obs.append(hdu.data)
                    self.obs_header.append(hdu.header)
        else:
            self.extract_from_db()

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
        '''GENERATE Z,T0,SALT2 PARAMS'''

        #Init randseed in order to reproduce SNs simulations
        if 'randseed' in self.sn_gen:
            self.randseed = int(self.sn_gen['randseed'])
        else:
            self.randseed = np.random.randint(low=1000,high=100000)

        randseeds = np.random.default_rng(self.randseed).integers(low=1000,high=100000,size=9)
        self.randseeds = {'z_seed': randseeds[0],
                          't0_seed': randseeds[1],
                          'x0_seed': randseeds[2],
                          'x1_seed': randseeds[3],
                          'c_seed': randseeds[4],
                          'coord_seed': randseeds[5],
                          'vpec_seed': randseeds[6],
                          'smearM_seed': randseeds[7],
                          'sigflux_seed': randseeds[8]
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

        if self.use_obs:
            self.extract_coord()
        else:
            self.db_to_obs()

        self.gen_z2cmb()
        self.gen_z_pec()
        self.zCMB = (1+self.zcos)*(1+self.zpec)-1.
        self.zobs = (1+self.zcos)*(1+self.zpec)*(1+self.z2cmb)-1.

        #SALT2 params generation
        self.gen_sn_par()
        self.gen_sn_mag()

        #self.sim_t0=np.zeros(self.n_sn)
        #Total fake for the moment....
        #self.sim_t0=np.array([52000+20+30*i for i in range(self.n_sn)])
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

    def extract_coord(self):
        '''Extract ra and dec from obs file'''
        # extract ra dec from obs config
        self.ra = []
        self.dec = []
        for i in range(self.n_sn):
            obs=self.obs_header[i]
            self.ra.append(obs['RA'])
            self.dec.append(obs['DEC'])
        return

    def gen_coord(self,randseeds):
        '''Generate ra,dec uniform on the sphere'''
        ra_seed = randseeds[0]
        dec_seed = randseeds[1]
        ra = np.random.default_rng(ra_seed).uniform(low=0,high=2*np.pi)
        dec_uni = np.random.default_rng(dec_seed).random()
        dec = np.arcsin(2*dec_uni-1)
        return ra, dec

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
        self.sim_x0 = mB_to_x0(self.sim_mB)
        return

    def gen_flux(self):
        ''' Generate simulated flux '''
        lc_seeds = np.random.default_rng(self.randseeds['sigflux_seed']).integers(low=1000,high=100000,size=self.n_sn)
        self.sim_lc=[]
        for obs,params,s in zip(self.obs,self.params,lc_seeds):
            lc = snc.realize_lcs(obs, self.model, [params], scatter=False)[0]
            lc['flux'] = np.random.default_rng(s).normal(loc=lc['flux'],scale=lc['fluxerr'])
            self.sim_lc.append(lc)

        return

    def plot_lc(self,lc_id,zp=25.,mag=False,plot_sim=True,plot_fit=False,residuals=False):
        '''Ploting the ligth curve of number 'lc_id' sn'''
        if plot_sim:
            sim_model = self.model
        else:
            sim_model = None

        if plot_fit:
            if self.fit_res[lc_id] == 'No_fit':
                raise ValueError("This lc wasn't fitted")
            fit_model = self.fit_res[lc_id][1]
            fit_cov = self.fit_res[lc_id][0]['covariance'][1:,1:]
        else:
            fit_model = None
            fit_cov = None
        plot_lc(self.sim_lc[lc_id],zp=zp,mag=mag,sim_model=self.model,fit_model=fit_model,fit_cov=fit_cov,residuals=residuals)
        return

    def fitter(self,id):
        try :
            res = snc_fit(self.sim_lc[id],self.model)
        except (RuntimeError):
            self.fit_res[id] = 'NaN'
            return
        self.fit_res[id] = res
        return

    def fit_lc(self,lc_id=None):
        '''Use sncosmo to fit sim lc'''
        if lc_id is None:
            for i in range(self.n_sn):
                self.model.set(z=self.sim_lc[i].meta['z'])  # set the model's redshift.
                self.fitter(i)
        else:
            self.model.set(z=self.sim_lc[lc_id].meta['z'])
            self.fitter(lc_id)
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

    def db_to_obs(self):
        '''Use a cadence db file to produce obs :
                1- Generate SN ra,dec in cadence fields
                2- Generate SN t0 in the survey time
                3- For each t0,ra,dec select visits that match
                4- Capture the information (filter, noise) of these visits
                5- Create sncosmo obs Table
         '''
        field_size=np.radians(np.sqrt(47)/2)
        ra_seeds, dec_seeds, choice_seeds = np.random.default_rng(self.randseeds['coord_seed']).integers(low=1000,high=10000,size=(3,self.n_sn))
        t0_seeds = np.random.default_rng(self.randseeds['t0_seed']).integers(low=1000,high=100000,size=self.n_sn)
        self.ra=[]
        self.dec=[]
        self.sim_t0=[]
        self.n_gen = 0
        for i in range(self.n_sn):
            compt = 0
            re_gen = True
            while re_gen:
                self.n_gen+=1
                #Gen ra and dec
                ra, dec = self.gen_coord([ra_seeds[i],dec_seeds[i]])
                #Gen t0
                t0 = np.random.default_rng(t0_seeds[i]).uniform(np.min(self.obs_dic['expMJD']),np.max(self.obs_dic['expMJD']))

                #Epochs selection
                ModelMinT_obsfrm = self.model.mintime()*(1+self.zcos[i])
                ModelMaxT_obsfrm = self.model.maxtime()*(1+self.zcos[i])
                epochs_selec =  (self.obs_dic['fieldRA']-field_size < ra)*(self.obs_dic['fieldRA']+field_size > ra) #ra selection
                epochs_selec *= (self.obs_dic['fieldDec']-field_size < dec)*(self.obs_dic['fieldDec']+field_size > dec) #dec selection
                epochs_selec *= (self.obs_dic['fiveSigmaDepth']>0) #use to avoid 1e43 errors
                epochs_selec *= (self.obs_dic['expMJD'] - t0  > ModelMinT_obsfrm)*(self.obs_dic['expMJD'] - t0 < ModelMaxT_obsfrm)

                #Cut on epochs
                for cut in self.nep_cut:
                    cutMin_obsfrm, cutMax_obsfrm = cut[1]*(1+self.zcos[i]), cut[2]*(1+self.zcos[i])
                    test = epochs_selec*(self.obs_dic['expMJD']-t0 > cutMin_obsfrm)
                    test *= (self.obs_dic['expMJD']-t0 < cutMax_obsfrm)
                    if len(cut) == 4:
                        test *= (np.vectorize(self.band_dic.get)(self.obs_dic['filter']) == cut[3])
                    if np.sum(test) < int(cut[0]):
                        re_gen = True
                        break
                    else:
                        re_gen = False
                if re_gen:
                    ra_seeds[i] = np.random.default_rng(ra_seeds[i]).integers(1000,100000)
                    dec_seeds[i] = np.random.default_rng(dec_seeds[i]).integers(1000,100000)

                if compt > len(self.obs_dic['expMJD']):
                    raise RuntimeError('Too many nep required, reduces nep_cut')
                else:
                    compt+=1


            self.ra.append(ra)
            self.dec.append(dec)
            self.sim_t0.append(t0)

            #Capture noise and filter
            mlim5 = self.obs_dic['fiveSigmaDepth'][epochs_selec]
            filter = self.obs_dic['filter'][epochs_selec].astype('U27')

            #Change band name to correpond with sncosmo bands
            if self.band_dic is not None:
                for i,f in enumerate(filter):
                    filter[i] = self.band_dic[f]
            #Convert maglim to flux noise (ADU)
            skynoise = pw(10.,0.4*(self.zp-mlim5))/5

            #Create obs table
            obs = Table({'time': self.obs_dic['expMJD'][epochs_selec],
                        'band': filter,
                        'gain': [self.gain]*np.sum(epochs_selec),
                        'skynoise': skynoise,
                        'zp': [self.zp]*np.sum(epochs_selec),
                        'zpsys': ['ab']*np.sum(epochs_selec)})
            self.obs.append(obs)
        self.ra = np.asarray(self.ra)
        self.dec = np.asarray(self.dec)
        return

    def extract_from_db(self):
        '''Read db file and extract relevant information'''

        dbf = sqlite3.connect(self.db_file)
        self.obs_dic={}
        keys=['expMJD', 'filter', 'fieldRA','fieldDec','fiveSigmaDepth','moonPhase']
        for k in keys:
            sql_com = f'SELECT {k} from Summary;'
            values = dbf.execute(sql_com)
            self.obs_dic[k] = np.array([a[0] for a in values])
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

    def plot_lc(self,lc_id,zp=25.,mag=False,fit=True):
        if fit:
            plot_lc(self.lc[lc_id],zp=zp,mag=mag,fit_model=self.fit_res[lc_id][1])
        else:
            plot_lc(self.lc[lc_id],zp=zp,mag=mag,sim_model=self.model)
