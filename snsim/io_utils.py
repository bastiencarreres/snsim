"""This module contains io stuff."""

import os
import warnings
import pickle
import pandas as pd
import numpy as np
try:
    import json
    imp_json = True
except ImportError:
    imp_json = False
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    imp_pyarrow = True
except ImportError:
    imp_pyarrow = False
try:
    import fastparquet as fq
    imp_fq = True
except ImportError:
    imp_fq = False


from . import salt_utils as salt_ut


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

            pickle.dump(pkl_dic, file)

    if 'parquet' in formats and imp_pyarrow and imp_json:
        lcs = pa.Table.from_pandas(data)
        lcmeta = json.dumps({int(data.attrs[k]['ID']): data.attrs[k] for k in data.attrs}, 
                             cls=NpEncoder)
        header = json.dumps(header, cls=NpEncoder)
        meta = {'name'.encode(): name.encode(),
                'attrs'.encode(): lcmeta.encode(),
                'header'.encode(): header.encode()}
        lcs = lcs.replace_schema_metadata(meta)
        pq.write_table(lcs, wpath + name + '.parquet')

    elif 'parquet' in formats and not imp_pyarrow and not imp_json:
        warnings.warn('You need pyarrow and json modules to use .parquet format', UserWarning)


def read_sim_file(file_path, engine='pyarrow'):
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
            lcs.index.set_names(['ID', 'epochs'], inplace=True)
            lcs.attrs = pkl_dic['meta']
            name = pkl_dic['name']
            header = pkl_dic['header']

    elif file_ext == '.parquet':
        if not imp_json:
            warnings.warn("You need json module to read parquet formats", UserWarning)
        if engine=='pyarrow' and imp_pyarrow:
            table = pq.read_table(file_path + file_ext)
            hdic = table.schema.metadata
            name = hdic['name'.encode()].decode()
            attrs_key = 'attrs'.encode()
            header_key = 'header'.encode()
        elif engine=='fastparquet' and imp_fq:
            table = fq.ParquetFile(file_path + file_ext)
            hdic = table.key_value_metadata
            name = hdic['name']
            attrs_key = 'attrs'
            header_key = 'header'
        elif not imp_pyarrow and not imp_fq:
            warnings.warn("You need pyarrow or fastparquet and json module to read parquet formats", UserWarning)
        lcs = table.to_pandas()
        lcs.set_index(['ID', 'epochs'], inplace=True)
        lcs.attrs = {int(k): val
                     for k, val in json.loads(hdic[attrs_key]).items()}
        header = json.loads(hdic[header_key])
    return name, header, lcs


def write_fit(sim_lcs_meta, fit_res, fit_model_name, sim_header, directory):
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
    fit_keys = ['t0', 'e_t0',
                'chi2', 'ndof']

    if fit_model_name in ('salt2', 'salt3'):
        fit_keys += ['x0', 'e_x0', 'mb', 'e_mb', 'x1',
                     'e_x1', 'c', 'e_c', 'cov_x0_x1', 'cov_x0_c',
                     'cov_mb_x1', 'cov_mb_c', 'cov_x1_c']
    
    data = {}
    for obj_ID in fit_res:
        snc_out = fit_res[obj_ID]['snc_out']
        data[obj_ID] = {**sim_lcs_meta[obj_ID]}
        data[obj_ID].pop('type')
        if snc_out != 'NaN':
            data[obj_ID] = {**data[obj_ID], **fit_res[obj_ID]['params']}
            data[obj_ID].pop('z')
            
            data[obj_ID]['e_t0'] = np.sqrt(snc_out['covariance'][0, 0])
        
            if MName[:5] in ('salt2', 'salt3'):
                par_cov = snc_out['covariance'][1:, 1:]
                mb_cov = salt_ut.cov_x0_to_mb(data[obj_ID]['x0'], par_cov)
                data[obj_ID]['e_x0'] = np.sqrt(par_cov[0, 0])
                data[obj_ID]['e_mb'] = np.sqrt(mb_cov[0, 0])
                data[obj_ID]['e_x1'] = np.sqrt(par_cov[1, 1])
                data[obj_ID]['e_c'] = np.sqrt(par_cov[2, 2])
                data[obj_ID]['cov_x0_x1'] = par_cov[0, 1]
                data[obj_ID]['cov_x0_c'] = par_cov[0, 2]
                data[obj_ID]['cov_x1_c'] = par_cov[1, 2]
                data[obj_ID]['cov_mb_x1'] = mb_cov[0, 1]
                data[obj_ID]['cov_mb_c'] = mb_cov[0, 2]

            data[obj_ID]['chi2'] = snc_out['chisq']
            data[obj_ID]['ndof'] = snc_out['ndof']
            data[obj_ID]['status'] = snc_out['success']
        else:
            for k in fit_keys:
                data[obj_ID][k] = np.nan
    df = pd.DataFrame.from_dict(data, orient='index')

    df.sort_values('ID', inplace=True)
    
    table = pa.Table.from_pandas(df, preserve_index=False)
    header = json.dumps(sim_header, cls=NpEncoder)
    table = table.replace_schema_metadata({'header'.encode(): header.encode()})
    pq.write_table(table, directory + '.parquet')
    print(f"Fit result output file : {directory}.parquet")


def open_fit(file):
    """USe to open fit file.

    Parameters
    ----------
    file : str
        Fit results parquet file
    Returns
    -------
    pandas.DataFrame
        The fit results.

    """
    table = pq.read_table(file)
    fit = table.to_pandas()
    fit.attrs = json.loads(table.schema.metadata['header'.encode()])
    fit.set_index(['ID'], inplace=True)

    return fit


def _read_sub_field_map(size, field_config):
    """Read the sub-field map file.

    Parameters
    ----------
    field_config : str
        Path to the field map config file.

    Returns
    -------
    dict
        A dict containing the corner postion of the field.

    """
    file = open(field_config)
    # Header symbol
    dic_symbol = {}
    nbr_id = -2
    lines = file.readlines()
    for i, l in enumerate(lines):
        if l[0] == '%':
            key_val = l[1:].strip().split(':')
            dic_symbol[key_val[0]] = {'nbr': nbr_id}
            dic_symbol[key_val[0]]['size'] = np.radians(float(key_val[2]))
            dic_symbol[key_val[0]]['type'] = key_val[1].lower()
            if key_val[1].lower() not in ['ra', 'dec']:
                raise ValueError('Espacement type is ra or dec')
            nbr_id -= 1
        else:
            break

    # Compute void region
    # For the moment only work with regular grid
    subfield_map = [string.strip().split(':') for string in lines[i:] if string != '\n']
    used_ra = len(subfield_map[0])
    used_dec = len(subfield_map)
    ra_space = 0
    for k in dic_symbol.keys():
        if dic_symbol[k]['type'] == 'ra':
            ra_space += subfield_map[0].count(k) * dic_symbol[k]['size']
            used_ra -= subfield_map[0].count(k)
    dec_space = 0
    for lines in subfield_map:
        if lines[0] in dic_symbol.keys() and dic_symbol[lines[0]]['type'] == 'dec':
            dec_space += dic_symbol[lines[0]]['size']
            used_dec -= 1

    subfield_ra_size = (size[0] - ra_space) / used_ra
    subfield_dec_size = (size[1] - dec_space) / used_dec

    # Compute all ccd corner
    corner_dic = {}
    dec_metric = size[1] / 2
    for i, l in enumerate(subfield_map):
        if l[0] in dic_symbol and dic_symbol[l[0]]['type'] == 'dec':
            dec_metric -= dic_symbol[l[0]]['size']
        else:
            ra_metric = - size[0] / 2
            for j, elmt in enumerate(l):
                if elmt in dic_symbol.keys() and dic_symbol[elmt]['type'] == 'ra':
                    ra_metric += dic_symbol[elmt]['size']
                elif int(elmt) == -1:
                    ra_metric += subfield_ra_size
                else:
                    corner_dic[int(elmt)] = np.array([
                        [ra_metric, dec_metric],
                        [ra_metric + subfield_ra_size, dec_metric],
                        [ra_metric + subfield_ra_size, dec_metric - subfield_dec_size],
                        [ra_metric, dec_metric - subfield_dec_size]])
                    ra_metric += subfield_ra_size
            dec_metric -= subfield_dec_size
    return corner_dic