class SuperNovae():
    __slots__ = ['zcos','zCMB','zpec','vpec','z2cmb','ra','dec','model_par']

    def __init__(self,n_sn,rand_seed,host=None):
        randseeds = l
        self.gen_coord(randseeds[0])
        self.gen_zcos(randseeds[1])
        self.gen_z2cmb()
        self.gen_z_pec()
        self.zCMB = (1 + self.zcos) * (1 + self.zpec) - 1.
        self.zobs = (1 + self.zcos) * (1 + self.zpec) * (1 + self.z2cmb) - 1.
        return

    def gen_coord(self, rand_seed, size=1):
        '''Generate ra,dec uniform on the sphere'''
        ra_seed = randseeds[0]
        dec_seed = randseeds[1]
        self.ra = np.random.default_rng(ra_seed).uniform(
            low=0, high=2 * np.pi, size=size)
        dec_uni = np.random.default_rng(dec_seed).random(size=size)
        self.dec = np.arcsin(2 * dec_uni - 1)
        return

    def gen_zcos(self,z_range,rand_seed):
        '''Function to get zcos, to be updated'''
        self.zcos = np.random.default_rng(rand_seed).uniform(
            low=z_range[0], high=z_range[1], size=size)
        return

    def gen_z2cmb(self):
        # use ra dec to simulate the effect of our motion
        coordfk5 = SkyCoord(
            self.ra * u.rad,
            self.dec * u.rad,
            frame='fk5')  # coord in fk5 frame
        galac_coord = coordfk5.transform_to('galactic')
        self.ra_gal = galac_coord.l.rad - 2 * np.pi * \
            np.sign(galac_coord.l.rad) * (abs(galac_coord.l.rad) > np.pi)
        self.dec_gal = galac_coord.b.rad

        ss = np.sin(self.dec_gal) * np.sin(self.dec_cmb * np.pi / 180)
        ccc = np.cos(self.dec_gal) * np.cos(self.dec_cmb * np.pi / \
                     180) * np.cos(self.ra_gal - self.ra_cmb * np.pi / 180)
        self.z2cmb = (1 - self.v_cmb * (ss + ccc) / su.c_light_kms) - 1.
        return

    def gen_z_pec(self,rand_seed,host=None):
        if host=None:
            self.vpec = np.random.default_rng(rand_seed).normal(
                loc=self.mean_vpec,
                scale=self.sig_vpec,
                size=self.n_sn)

            if 'vp_sight' in self.host.names:
                self.vpec = self.host['vp_sight']
        else:

        self.zpec = self.vpec / su.c_light_kms
        return

class ObsTable():
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
        return epochs_selection
