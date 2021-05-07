class SN:
    __slots__ = ['zcos','zpec','zCMB','ra','dec','vpec','t0']

    def __init__(self,zcos,ra,dec,vpec,t0,sim_mu,cosmo,mag_smear,sim_model,model_par):
        self.sim_t0 = t0
        self.zcos = zcos
        self.z2cmb = z2cmb
        self.ra = ra
        self.dec = dec
        self.vpec = vpec
        self.sim_mu = sim_mu
        self.mag_smear = mag_smear
        self.model_par = model_par
        self.sim_model = model.__copy__()
        self._epochs = None
        self.sim_lc = None
        return

    @property
    def zpec(self):
        return self.vpec/su.c_light_kms

    @property
    def zCMB(self):
        return (1+self.zcos)*(1+self.zpec) - 1.

    @property
    def z(self):
        return (1+self.zcos)*(1+self.zpec)*(1+self.z2cmb) - 1.

    @property
    def sim_mb(self):
        # Compute mB : { mu + M0 : the standard magnitude} + {-alpha*x1 +
        # beta*c : scattering due to color and stretch} + {intrinsic smearing}
        return self.sim_mu + self.M0 - self.alpha * \
        self.sim_x1 + self.beta * self.sim_c + self.mag_smear

    @property
    def sim_x0(self):
        return su.mB_to_x0(self.sim_mB)

    @property
    def epochs(self):
         return self._epochs

    @epochs.setter
    def epochs(self,ep_dic):
        self._epochs = ep_dic


    def z2cmb(self):
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

class SNGen:
    def __init__(self,n_sn,z_range,sigmaM,rand_seed,host=None):
        self.n_sn = n_sn
        self.z_range = z_range
        randseeds = np.random.default_rng(rand_seed).randint(low=1000, high=100000,size=6)
        self.rand_seed = {'z' : randseeds[0],
                          't0': randseeds[1],
                          'coord': randseed[2],
                          'coh_scatter': randseeds[3],
                          'mod_scatter': randseed[4],
                          'sn_model': randseed[5]
                         }

        self.sigmaM = sigmaM
        self.sim_model = sim_model

        return

    def __call__(self):
        if sel.host == None:
            ra, dec = self.gen_coord()
            zcos = self.gen_zcos(se)
            z2cmb = self.gen_z2cmb(ra,dec)
            vpec, zpec = self.gen_zpec(host)

        mag_smear = self.gen_coh_scatter(self.sigmaM)

        sn_par = [{'zcos': z,
                   't0': t,
                   'ra': r,
                   'dec': d,
                   'vpec': v,
                   'mag_smear': ms
                    } for z,t,r,d,v,ms in zip(zcos,t0,ra,dec,mag_smear)]

        model
        SN_list = [SN(z1,v,r,d,t,ms,mod) for z1,z2,v,r,d,t,mu,ms,mod in zip(zcosvpec,ra,dec,t0,sim_mu,mag_smear,model)]
        return SN_list

    def gen_coord(self, rand_seed, size=1):
        '''Generate ra,dec uniform on the sphere'''
        ra_seed = rand_seed[0]
        dec_seed = rand_seed[1]
        ra = np.random.default_rng(ra_seed).uniform(
            low=0, high=2 * np.pi, size=size)
        dec_uni = np.random.default_rng(dec_seed).random(size=size)
        dec = np.arcsin(2 * dec_uni - 1)
        return ra, dec

    def gen_zcos(self):
        '''Function to get zcos, to be updated'''
        zcos = np.random.default_rng(rand_seed).uniform(
            low=self.z_range[0], high=self.z_range[1], size=size)
        return zcos

    def gen_z2cmb(ra,dec):
        # use ra dec to simulate the effect of our motion
        coordfk5 = SkyCoord(
            ra * u.rad,
            dec * u.rad,
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

    def gen_model_param(self):
        if sim_model.source.name in ['salt2','salt3']:
            self.gen_SALT_par()

        if 'G10_' in sim_model.effect_name:
            self.model_par['G10_RndS'] = self.rand_seed['mod_scatter']

        elif 'C11_' in sim_model.effect_name:
            self.model_par['C11_RndS'] = self.rand_seed['mod_scatter']
        return

    def gen_SALT_par(self):
        ''' Generate x1 and c for the SALT2 or SALT3 model'''
            x1_seed,c_seed = np.random.default_rng(self.rand_seed['sn_model']).randint(low=1000, high=100000,size=2)
            sim_x1 = np.random.default_rng(x1_seed).normal(
            loc=x1_distrib[0],
            scale=x1_distrib[1],
            size=self.n_sn)
            sim_c = np.random.default_rng(c_seed).normal(
            loc=c_distrib[0], scale=c_distrib[1], size=self.n_sn)
            self.model_par['x1'] = sim_x1
            self.model_par['c'] = sim_c
            return

    def gen_zpec(self,rand_seed,host=None):
        if host=None:
            vpec = np.random.default_rng(rand_seed).normal(
                loc=self.mean_vpec,
                scale=self.sig_vpec,
                size=self.n_sn)

            if 'vp_sight' in self.host.names:
                self.vpec = self.host['vp_sight']
        else:
            vpec = host.vpec
        return vpec

    def gen_coh_scatter(self):
        ''' Generate x0/mB parameters for SALT2 '''
        mag_smear = np.random.default_rng(self.rand_seed['coh_scatter']).normal(loc=0, scale=self.sigmaM, size=self.n_sn)
        return mag_smear

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
