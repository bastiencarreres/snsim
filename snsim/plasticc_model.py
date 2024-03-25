import os
import sncosmo as snc
from snsim import __snsim_dir_path__
import glob
import requests
import tarfile
import shutil


plasticc_repo = 'https://zenodo.org/records/6672739/'

model_repo = { 
            'slsn' : plasticc_repo + 'SIMSED.SLSN-I-MOSFIT.tar.gz',
            'sniax' : plasticc_repo + 'SIMSED.SNIax.tar.gz',
            'snia91bg' : plasticc_repo + 'SISIMSED.SNIa-91bg.tar.gz'           
             }

def check_files_and_download(model_name):
    """Check if model files are here and download from Plasticc repository if not. 
    availabele model are SLSN, SNIax, SNIa91bg

    Returns
    -------
    None
        No return, just download files.
    
    Notes
    -----
    TODO : Change that for environement variable or cleaner solution

    """
    
    data_dir_name = model_name.lower() + '_data'

    if  not os.path.isdir(_snsim_dir_path__ + data_dir_name + '/'):
        print("Dowloading model template files files from ", model_repo[model_name.lower()])
        os.mkdir(snsim_dir_path__ + data_dir_name)
        url = model_repo[model_name.lower()]
        response = requests.get(url, stream=True)
        dir_tar = tarfile.open(fileobj=response.raw, mode="r|gz")
        for file in dir_tar.getmembers():
            file.extractall( path=_snsim_dir_path__ + data_dir_name )
        shutil.rmtree(snsim_dir_path_ + model_repo[model_name.lower()] )
        

def get_sed_listname(model_name):

    check_files_and_download(model_name)
    data_dir_name = model_name.lower() + '_data'

    file_list=[]
    for file in os.listdir(snsim_dir_path__ + data_dir_name):
        name = filename.replace('.SED','')
        file_list.append(name)

    return file_list
    


def sncosmo_model_from_SED(model):

    import numpy as np

    phase = np.linspace(-50., 50., 11)

    disp = np.linspace(3000., 8000., 6)

    flux = np.repeat(np.array([[0.], [1.], [2.], [3.], [4.], [5.],

                            [4.], [3.], [2.], [1.], [0.]]),

                    6, axis=1)

    source = sncosmo.TimeSeriesSource(phase, disp, flux)

    #the file are already well organized, just ask rick the unit of the flux
        #create sncosmo model from it 

    model = sncosmo.Model(source)
        # init the model



