import os
import glob
import requests
import tarfile
import sncosmo as snc
import numpy as np
import scipy.stats as stats
from snsim import __snsim_dir_path__


plasticc_repo = 'https://zenodo.org/records/6672739/files/'
PlasticcDir = snc.utils.DataMirror(snc.builtins.get_rootdir, "")
PlasticcDir._redirects = {
    'models/plasticc/SIMSED.SNIa-91bg.tar.gz': plasticc_repo + 'SIMSED.SNIa-91bg.tar.gz',
    'models/plasticc/SIMSED.SNIax.tar.gz': plasticc_repo + 'SIMSED.SNIax.tar.gz',
    'models/plasticc/SIMSED.SLSN-I-MOSFIT.tar.gz': plasticc_repo + 'SIMSED.SLSN-I-MOSFIT.tar.gz'
    }

def load_plasticc_timeseries(relpath, fname, zero_before=True, time_spline_degree=3,
                             name=None, version=None):
    abspath = PlasticcDir.abspath(relpath, isdir=True)
    fpath = abspath + '/' + fname
    if os.path.isfile(fpath + '.gz'):
        import gzip
        import shutil

        with gzip.open(fpath + '.gz', 'rb') as f_in:
            with open(fpath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(fpath + '.gz')

    phase, wave, flux = snc.io.read_griddata_ascii(fpath)
    return snc.models.TimeSeriesSource(phase, wave, flux, 
                            name=name, version=version,
                            zero_before=zero_before,
                            time_spline_degree=time_spline_degree)

# Add to sncosmo registry
plasticc_SN91bg = [
    (f'plasticc-snia-91bg-st{i}_c{j}', 'SN Ia-91bg', 
    f'91BG_ST{i}_C{j}.SED', 'models/plasticc/SIMSED.SNIa-91bg')
    for i in range(6) for j in range(5)]

plasticc_SNIax = [
    (f'plasticc-sniax-{i:04d}', 'SN Iax', 
    f'SED-Iax-{i:04d}.dat', 'models/plasticc/SIMSED.SNIax') 
    for i in range(1001)
]


# TO TEST
#plasticc_SLSN = [
   ## (f'plasticc-slsn-{i}', 'SLSN', 
    #f'slsn{i}.dat', 'models/plasticc/SIMSED.SLSN-I-MOSFIT') 
    #for i in range(1000)
#]

plasticc = plasticc_SN91bg + plasticc_SNIax 

ref = ('TBD', 'TBD', 'TBD')
for name, sntype, fn, relpath in plasticc:
    meta = {
        'dataurl': plasticc_repo, 
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'type': sntype,
        'ref': ref}

    snc.models._SOURCES.register_loader(
        name, load_plasticc_timeseries,
        args=(relpath, fn), version=None, meta=meta)


def generate_dust_sniax(n_sn, seed=None):
    rand_gen = np.random.default_rng(seed)
    lower, upper = 0.5, 10000
    mu, sigma = 2, 1.4
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    Rv = X.rvs(n_sn)
    E_dust = rand_gen.exponential(scale=0.1, size=n_sn)
    return Rv, E_dust

def get_sed_listname(model_name):
    sources = snc.builtins._SOURCES.get_loaders_metadata()

    if model_name == "sniax":
        return list(s["name"] for s in sources if "sn iax" in s["type"].lower())

    if model_name == "snia91bg":
         return list(s["name"] for s in sources if "sn ia-91bg" in s["type"].lower())

    if model_name == "slsn":
        return list(s["name"] for s in sources if "slsn" in s["type"].lower())

# # TO DO: Remove
# model_repo = {
#     "slsn": plasticc_repo + "SIMSED.SLSN-I-MOSFIT.tar.gz",
#     "sniax": plasticc_repo + "SIMSED.SNIax.tar.gz",
#     "snia91bg": plasticc_repo + "SIMSED.SNIa-91bg.tar.gz",
# }

# def snc_source_from_sed(path, name=None):
#     phase_sed, wave_sed, flux_sed = np.genfromtxt(path, unpack=True)
#     phase = np.unique(phase_sed)
#     disp = np.unique(wave_sed)
#     flux = []
#     for p in np.unique(phase_sed):
#         idx = np.where(phase_sed == p)
#         flux.append(flux_sed[idx])

#     flux = np.asarray(flux)
#     source = snc.TimeSeriesSource(phase, disp, flux, name=name, version="plasticc")
#     return source


# def snc_model_from_sed(filename, path, eff, eff_names, eff_frames):
#     source = snc_source_from_sed(filename, path)

#     model = snc.Model(
#         source=source,
#         effects=eff,
#         effect_names=eff_names,
#         effect_frames=eff_frames,
#     )
#     return model


# def check_files_and_download(model_name):
#     """Check if model files are here and download from Plasticc repository if not.
#     availabele model are SLSN, SNIax, SNIa91bg

#     Returns
#     -------
#     None
#         No return, just download files.

#     Notes
#     -----
#     TODO : Change that for environement variable for cleaner solution

#     """

#     data_dir_name = "sed_data/" + model_name.lower() + "_data"

#     if not os.path.isdir(__snsim_dir_path__ + "/" + data_dir_name + "/"):
#         print(
#             "Dowloading model template files files from ",
#             model_repo[model_name.lower()],
#         )
#         os.makedirs(__snsim_dir_path__ + "/" + data_dir_name + "/")
#         url = model_repo[model_name.lower()]
#         print(url)
#         response = requests.get(url, stream=True)
#         dir_tar = tarfile.open(fileobj=response.raw, mode="r|gz")
#         dir_tar.extractall(path=__snsim_dir_path__ + "/" + data_dir_name + "/")
