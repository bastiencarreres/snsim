"""This module contains io stuff."""

import os
import warnings
import pickle
import pandas as pd
import numpy as np
try:
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    json_pyarrow = True
except ImportError:
    json_pyarrow = False
from . import salt_utils as salt_ut
from astropy.table import Table
from astropy.io import fits


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def write_sim(wpath, name, formats, header, data):
    """Write simulated lcs.

    Parameters
    ----------
    wpath : str
        The path where to write file.
    name : str
        Simulation name.
    formats : np.array(str)
        List of files fopprmats to write.
    header : dict
        The simulation header.
    data : pandas.DataFrame
        Dataframe containing lcs.

    Returns
    -------
    None
        Just write files.

    """
    # Export lcs as pickle
    if 'pkl' in formats:
        with open(wpath + name + '.pkl', 'wb') as file:
            pkl_dic = {'name': name,
                       'lcs': data.to_dict(),
                       'meta': data.attrs,
                       'header': header}

            pickle.dump(pkl_dic,
                        file)

    if 'parquet' in formats and json_pyarrow:
        lcs = pa.Table.from_pandas(data)
        lcmeta = json.dumps(data.attrs, cls=NpEncoder)
        header = json.dumps(header, cls=NpEncoder)
        meta = {'name'.encode(): name.encode(),
                'attrs'.encode(): lcmeta.encode(),
                'header'.encode(): header.encode()}
        lcs = lcs.replace_schema_metadata(meta)
        pq.write_table(lcs, wpath + name + '.parquet')

    elif 'parquet' in formats and not json_pyarrow:
        warnings.warn('You need pyarrow and json modules to use .parquet format', UserWarning)

    # TO DO : Re-implement fits format?
    # if 'fits' in formats:
    #     lc_hdu_list = (fits.table_to_hdu(lc) for lc in lcs_list)
    #     hdu_list = fits.HDUList(
    #         [fits.PrimaryHDU(header=fits.Header(header))] + list(lc_hdu_list))
    #
    #     hdu_list.writeto(write_path + self.name + sufname + '.fits',
    #                      overwrite=True)


def read_sim_file(file_path):
    """Read a sim file.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    str, dict, pandas.DataFrame
        The name, the header and the lcs of the simulation.

    """
    file_path, file_ext = os.path.splitext(file_path)

    if file_ext == '.pkl':
        with open(file_path + file_ext, 'rb') as f:
            pkl_dic = pickle.load(f)
            lcs = pd.DataFrame.from_dict(pkl_dic['lcs'])
            lcs.index.set_names(['sn_id', 'epochs'], inplace=True)
            lcs.attrs = pkl_dic['meta']
            name = pkl_dic['name']
            header = pkl_dic['header']

    elif file_ext == '.parquet':
        if json_pyarrow:
            table = pq.read_table(file_path + file_ext)
            lcs = table.to_pandas()
            lcs.set_index(['sn_id', 'epochs'], inplace=True)
            lcs.attrs = {int(k): val for k, val in json.loads(table.schema.metadata['attrs'.encode()]).items()}
            name = table.schema.metadata['name'.encode()].decode()
            header = json.loads(table.schema.metadata['header'.encode()])
        else:
            warnings.warn("You need pyarrow and json module to write parquet formats", UserWarning)

        # TO DO : Re-implement fits ?
        # sample_name = os.path.basename(file_path)
        # if file_ext == '.fits':
        #     with fits.open(file_path + file_ext) as sf:
        #         sim_lcs = []
        #         for i, hdu in enumerate(sf):
        #             if i == 0:
        #                 header = hdu.header
        #             else:
        #                 tab = hdu.data
        #                 tab.meta = hdu.header
        #                 sim_lcs.append(tab)
        #     return cls(sample_name, sim_lcs, header, model_dir=model_dir,
        #                dir_path=os.path.dirname(file_path) + '/')

    return name, header, lcs


def write_fit(sim_lcs_meta, fit_res, fit_dic, directory, sim_meta={}):
    """Write fit into a fits file.

    Parameters
    ----------
    sim_lcs_meta : dict{list}
        Meta data of all lightcurves.
    fit_res : list(sncosmo.utils.Result)
        List of sncosmo fit results for each lightcurve.
    directory : str
        Destination of write file.
    sim_meta : dict
        General simulation meta data.

    Returns
    -------
    None
        Just write a file.

    """
    data = sim_lcs_meta.copy()

    fit_keys = ['t0', 'e_t0',
                'chi2', 'ndof']
    MName = sim_meta['model_name']

    if MName[:5] in ('salt2', 'salt3'):
        fit_keys += ['x0', 'e_x0', 'mb', 'e_mb', 'x1',
                     'e_x1', 'c', 'e_c', 'cov_x0_x1', 'cov_x0_c',
                     'cov_mb_x1', 'cov_mb_c', 'cov_x1_c']

    for k in fit_keys:
        data[k] = []

    for res, fd in zip(fit_res, fit_dic):
        if res != 'NaN':
            par = res['parameters']
            data['t0'].append(fd['t0'])
            data['e_t0'].append(np.sqrt(res['covariance'][0, 0]))

            if MName[:5] in ('salt2', 'salt3'):
                par_cov = res['covariance'][1:, 1:]
                mb_cov = salt_ut.cov_x0_to_mb(par[2], par_cov)
                data['x0'].append(fd['x0'])
                data['e_x0'].append(np.sqrt(par_cov[0, 0]))
                data['mb'].append(fd['mb'])
                data['e_mb'].append(np.sqrt(mb_cov[0, 0]))
                data['x1'].append(fd['x1'])
                data['e_x1'].append(np.sqrt(par_cov[1, 1]))
                data['c'].append(fd['c'])
                data['e_c'].append(np.sqrt(par_cov[2, 2]))
                data['cov_x0_x1'].append(par_cov[0, 1])
                data['cov_x0_c'].append(par_cov[0, 2])
                data['cov_x1_c'].append(par_cov[1, 2])
                data['cov_mb_x1'].append(mb_cov[0, 1])
                data['cov_mb_c'].append(mb_cov[0, 2])

            data['chi2'].append(res['chisq'])
            data['ndof'].append(res['ndof'])
        else:
            for k in fit_keys:
                data[k].append(np.nan)

    for k, v in sim_lcs_meta.items():
        data[k] = v

    table = Table(data)

    hdu = fits.table_to_hdu(table)
    hdu_list = fits.HDUList([fits.PrimaryHDU(header=fits.Header(sim_meta)), hdu])
    hdu_list.writeto(directory, overwrite=True)
    print(f'Fit result output file : {directory}')
