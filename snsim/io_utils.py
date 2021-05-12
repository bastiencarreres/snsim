from . import utils as ut

def write_fit(sim_lc_meta, fit_res, directory,sim_meta={}):
    data = sim_lc_meta.copy()

    fit_keys = ['t0', 'e_t0',
                'chi2', 'ndof']
    MName = sim_meta['MName']
    if MName == 'salt2' or MName=='salt3':
        fit_keys += ['x0', 'e_x0', 'mb', 'e_mb', 'x1',
                     'e_x1', 'c', 'e_c', 'cov_x0_x1', 'cov_x0_c',
                     'cov_mb_x1', 'cov_mb_c', 'cov_x1_c']

    for k in fit_keys:
        data[k] = []

    for i in sim_lc_meta['sn_id']:
        if fit_res[i] != np.nan:
            par = fit_res[i]['parameters']
            data['t0'].append(par[1])
            data['e_t0'].append(np.sqrt(fit_res[i]['covariance'][0, 0]))

            if MName == 'salt2' or MName=='salt3':
                par_cov = fit_res[i]['covariance'][1:, 1:]
                mb_cov = ut.cov_x0_to_mb(par[2], par_cov)
                data['x0'].append(par[2])
                data['e_x0'].append(np.sqrt(par_cov[0, 0]))
                data['mb'].append(ut.x0_to_mB(par[2]))
                data['e_mb'].append(np.sqrt(mb_cov[0, 0]))
                data['x1'].append(par[3])
                data['e_x1'].append(np.sqrt(par_cov[1, 1]))
                data['c'].append(par[4])
                data['e_c'].append(np.sqrt(par_cov[2, 2]))
                data['cov_x0_x1'].append(par_cov[0, 1])
                data['cov_x0_c'].append(par_cov[0, 2])
                data['cov_x1_c'].append(par_cov[1, 2])
                data['cov_mb_x1'].append(mb_cov[0, 1])
                data['cov_mb_c'].append(mb_cov[0, 2])

            data['chi2'].append(fit_res[i]['chisq'])
            data['ndof'].append(fit_res[i]['ndof'])
        else:
            for k in fit_keys:
                data[k].append(np.nan)

    table = Table(data)

    hdu = fits.table_to_hdu(table)
    hdu_list = fits.HDUList([fits.PrimaryHDU(header=fits.Header(sim_meta)), hdu])
    hdu_list.writeto(directory, overwrite=True)
    print(f'Fit result output file : {directory}')
    return
