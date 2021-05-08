import numpy as np

class SNGen:
    def __init__(self,snc_sim_model,sim_par,host=None):
        self.sim_model = snc_sim_model
        self.model_par = sim_par['model_par']
        self.model_keys = ['M0']
        self.model_keys += self.init_model
        self.cosmo_dic = sim_par['cosmo_dic']
        self.host = host

    def isSalt(self):
        if self.sim_model.source.name in ['salt2','salt3']
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
        rand_seeds = np.random.default_rng(rand_seed).randint(low=1000, high=100000,size=6)
        ra, dec = self.gen_coord(n_sn,rand_seeds[0])
        zcos = self.gen_zcos(n_sn,rand_seeds[1])
        vpec = self.gen_vpec(n_sn,host,rand_seeds[2])
        mag_smear = gen_coh_scatter(n_sn,rand_seeds[3])

        sn_par = [{'zcos': z,
                   'sim_t0': t,
                   'ra': r,
                   'dec': d,
                   'vpec': v,
                   'mag_smear': ms
                   } for z,t,r,d,v,ms in zip(zcos,t0,ra,dec,mag_smear)]

        model_par_sncosmo = gen_model_param(n_sn,rand_seed[4:6])

        SN_list = [SN(snp,self.cosmo_dic,) for snp in zip(sn_par,ra,dec,t0,sim_mu,mag_smear,model)]
        return SN_list

    def gen_coord(rand_seed):
    '''Generate ra,dec uniform on the sphere'''
        coord_seed = np.random.default_rng(rand_seed).randint(low=1000,high=100000,size=2)
        ra = np.random.default_rng(coord_seed[0]).uniform(
            low=0, high=2 * np.pi, size=size)
        dec_uni = np.random.default_rng(coord_seed[1]).random(size=size)
        dec = np.arcsin(2 * dec_uni - 1)
        return ra, dec

    def gen_zcos(z_range, rand_seed):
    '''Function to get zcos, to be updated'''
        zcos = np.random.default_rng(rand_seed).uniform(
        low=z_range[0], high=z_range[1], size=size)
        return zcos

    def gen_sncosmo_param(self,rand_seed):
        if self.isSalt:
            sim_x1, sim_c = self.gen_SALT_par(rand_seed[0])
            model_par_sncosmo = [{'x1': x1, 'c': c} for x1, c in zip(sim_x1,sim_c)]

        if 'G10_' in self.sim_model.effect_name:
            seeds = np.random.default_rng(rand_seed[1]).integers(low=1000, high=100000,size=self.n_sn)
            for par,s in zip(model_par_sncosmo,seeds):
                model_par_sncosmo['G10_RndS'] = s

        elif 'C11_' in self.sim_model.effect_name:
            seeds = np.random.default_rng(rand_seed[1]).integers(low=1000, high=100000,size=self.n_sn)
            for par,s in zip(model_par_sncosmo,seeds):
                par['C11_RndS'] = s

        return model_par_sncosmo

    def gen_SALT_par(n_sn,rand_seed):
    ''' Generate x1 and c for the SALT2 or SALT3 model'''
        x1_seed,c_seed = np.random.default_rng(rand_seed).randint(low=1000, high=100000,size=2)
        sim_x1 = np.random.default_rng(x1_seed).normal(
        loc=x1_distrib[0],
        scale=x1_distrib[1],
        size=n_sn)
        sim_c = np.random.default_rng(c_seed).normal(
        loc=c_distrib[0], scale=c_distrib[1], size=n_sn)
        return sim_x1, sim_c

    def gen_vpec(self,rand_seed,host=None):
        if host == None:
            vpec = np.random.default_rng(rand_seed).normal(
                loc=self.mean_vpec,
                scale=self.sig_vpec,
                size=self.n_sn)

            if 'vp_sight' in self.host.names:
                self.vpec = self.host['vp_sight']
        else:
            vpec = host.vpec
        return vpec

    def gen_coh_scatter(self,n_sn,rand_seed):
        ''' Generate x0/mB parameters for SALT2 '''
        mag_smear = np.random.default_rng(rand_seed).normal(loc=0, scale=self.sigCOH, size=n_sn)
        return mag_smear
