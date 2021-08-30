"""Script to launch snsim in commande line."""

import argparse
import yaml
from snsim import Simulator


def assym_read(assym):
    """Read assym gaussian parameters."""
    if len(assym) == 1:
        assym = assym[0]
    return assym


def mw_dust_read(mw_dust):
    """Read the mw_dust argument."""
    if len(mw_dust) == 1:
        mw_dust = mw_dust[0]
    else:
        mw_dust[1] = float(mw_dust[1])
    return mw_dust


def date_read(date):
    """Read the date arguments."""
    if date.isdigit():
        return float(date)
    return date


def nep_cut(nepc):
    """Read the nep_cut arguments."""
    for i in range(len(nepc)):
        if len(nepc[i]) >= 1:
            nepc[i][0] = int(nepc[i][0])
        if len(nepc[i]) >= 3:
            nepc[i][1] = float(nepc[i][1])
            nepc[i][2] = float(nepc[i][2])
    return nepc


parser = argparse.ArgumentParser()
parser.add_argument(
    "config_path",
    default=None,
    type=str,
    help="Configuration file")
parser.add_argument("-fit", action='store_true')

keys_dic = {
    'data': [
        'write_path',
        'sim_name',
        'write_format'],
    'survey_config': [
        'survey_file',
        'band_dic',
        'db_cut',
        'zp',
        'sig_zp',
        'sig_psf',
        'noise_key',
        'gain',
        'ra_size',
        'dec_size',
        'start_day',
        'end_day',
        'duration',
        'sub_field',
        'field_map'],
    'sn_gen': [
        'randseed',
        'duration_for_rate',
        'n_sn',
        'sn_rate',
        'rate_pw',
        'nep_cut',
        'z_range',
        'M0',
        'mag_sct',
        'sct_mod'],
    'cosmology': [
        'Om0',
        'H0'],
    'cmb': [
        'v_cmb',
        'ra_cmb',
        'dec_cmb'],
    'model_config': [
        'model_dir',
        'model_name',
        'alpha',
        'beta',
        'mean_x1',
        'mean_c',
        'sig_x1',
        'sig_c',
        'mw_dust'],
    'vpec_dist': [
        'mean_vpec',
        'sig_vpec'],
    'alpha_dipole': [
        'coord',
        'A',
        'B']}

################
# DATA SECTION #
################
parser.add_argument("--write_path", type=str, help="'/PATH/TO/OUTPUT'")
parser.add_argument("--sim_name", type=str, help="'NAME OF SIMULATION'")
parser.add_argument("--write_format", type=str, nargs='+', help="'format' or ['format1','format2']")

#########################
# SURVEY_CONFIG SECTION #
#########################
parser.add_argument("--survey_file", type=str, help='/PATH/TO/FILE')
parser.add_argument("--band_dic", type=yaml.load,
                    help="{'band1_db_key':'band1_sncosmo_key', ...} database key -> sncosmo key,\
                           put dictionnary between double quotationmark")
parser.add_argument("--add_data", type=str, nargs='+',
                    help="--add_data 'key1' 'key2' ... # Add metadata from survey file to output")
parser.add_argument("--db_cut", type=yaml.load,
                    help="{'key1': ['conditon1','conditon2',...], 'key2':['conditon1'],...},\
                          put dictionnary between double quotation mark")
parser.add_argument("--zp", type=float, help="INSTRUMENTAL ZEROPOINT")
parser.add_argument("--sig_zp", type=float, help="UNCERTAINTY ON ZEROPOINT")
parser.add_argument("--sig_psf", type=float, help="GAUSSIAN PSF SIGMA")
parser.add_argument("--noise_key", type=str, nargs=2,
                    help="--noise_key key type, type can be 'mlim5' or 'skysigADU'")
parser.add_argument("--gain", type=float, help="CCD GAIN e-/ADU")
parser.add_argument("--ra_size", type=float, help="RA FIELD SIZE in DEG")
parser.add_argument("--dec_size", type=float, help="Dec FIELD SIZE in DEG")
parser.add_argument("--start_day", type=date_read,
                    help="Survey starting day MJD NUMBER or 'YYYY-MM-DD'")
parser.add_argument("--end_day", type=date_read,
                    help="Survey ending day MJD NUMBER or 'YYYY-MM-DD'")
parser.add_argument("--duration", type=float, help="SURVEY DURATION IN DAYS")
parser.add_argument("--sub_field", type=str, help="SUBFIELD KEY")
parser.add_argument("--field_map", type=str, help="SUBFIELD MAP")

###################
# SN_GEN SECTION #
##################
parser.add_argument("--randseed", type=int, help="RANDSEED TO REPRODUCE SIMULATION")
parser.add_argument("--duration_for_rate", type=float,
                    help="FAKE DURATION ONLY USED TO GENERATE N SN (DAYS)")
parser.add_argument("--n_sn", type=int, help="NUMBER OF SN TO GENERATE")
parser.add_argument("--sn_rate", type=float, help="rate of SN/Mpc^3/year")
parser.add_argument("--rate_pw", type=float, help="rate = sn_rate*(1+z)^rate_pw")
parser.add_argument("--nep_cut", action='append', nargs='+',
                    help="--nep_cut nep_min1 Tmin Tmax --nep_cut nep_min2 Tmin2 Tmax2 'filter1',\
                    put cuts on the number of epochs between Tmin and Tmax \
                    (restframe, relative to peak), optionaly in a selected filter")
parser.add_argument("--z_range", type=float, nargs=2,
                    help="--zrange zmin zmax, Cosmological redshift range")
parser.add_argument("--M0", type=float, help="SN ABSOLUT MAGNITUDE")
parser.add_argument("--mag_sct", type=float, help="SN INTRINSIC COHERENT SCATTERING")
parser.add_argument("--sct_mod", type=str,
                    help="'G10','C11_i' USE WAVELENGHT DEP MODEL FOR SN INT SCATTERING")

#####################
# COSMOLOGY SECTION #
#####################
parser.add_argument("--Om0", type=float, help="MATTER DENSITY")
parser.add_argument("--H0", type=float, help="HUBBLE CONSTANT")

###############
# CMB SECTION #
###############
parser.add_argument("--v_cmb", type=float, help="OUR PECULIAR VELOCITY")
parser.add_argument("--ra_cmb", type=float, help="GAL L OF CMB DIPOLE")
parser.add_argument("--dec_cmb", type=float, help="GAL B OF CMB DIPOLE")

########################
# MODEL_CONFIG SECTION #
########################
parser.add_argument("--model_dir", type=str, help="'/PATH/TO/SALT/MODEL'")
parser.add_argument("--model_name", type=int, help="'THE MODEL NAME', example 'salt2'")
parser.add_argument("--mw_dust", type=str, nargs='+',
                    help="--mw_dust 'MODEL_NAME' or --mw_dust ['MODEL_NAME', RV]")
# SALT PARAM
parser.add_argument("--alpha", type=float, help="STRETCH CORRECTION")
parser.add_argument("--beta", type=float, help="COLOR CORRECTION")
parser.add_argument("--mean_x1", type=float, help="MEAN X1 VALUE")
parser.add_argument("--sig_x1", type=float, nargs='+',
                    help="--sig_x1 sigma or --sigma_x1 sigma- sigma+")
parser.add_argument("--mean_c", type=float, help="MEAN C VALUE")
parser.add_argument("--sig_c", type=float, nargs='+',
                    help="--sig_c sigma or --sigma_c sigma- sigma+")

#####################
# VPEC_DIST SECTION #
#####################
parser.add_argument("--mean_vpec", type=float, help="MEAN SN PECULIAR VELOCITY")
parser.add_argument("--sig_vpec", type=float, help="SIGMA PECULIAR VELOCITY")

#####################
# HOST_FILE SECTION #
#####################
parser.add_argument("--host_file", type=str, help="'/PATH/TO/HOSTFILE'")

########################
# ALPHA_DIPOLE SECTION #
########################
parser.add_argument("--alpha_coord", dest="coord", type=float, nargs=2,
                    help="--alpha_coord RA Dec, Alpha dipole coordinates")
parser.add_argument("--alpha_A", dest="A", type=float,
                    help="Alpha dipole = A + B * cos(theta)")
parser.add_argument("--alpha_B", dest="B", type=float,
                    help="Alpha dipole = A + B * cos(theta)")

args = parser.parse_args()

if args.nep_cut is not None:
    args.nep_cut = nep_cut(args.nep_cut)

if args.mw_dust is not None:
    args.mw_dust = mw_dust_read(args.mw_dust)

if args.sig_x1 is not None:
    args.sig_x1 = assym_read(args.sig_x1)

if args.sig_c is not None:
    args.sig_c = assym_read(args.sig_c)

with open(args.config_path, "r") as f:
    yml_config = yaml.safe_load(f)

param_dic = {}
for K in keys_dic:
    for k in keys_dic[K]:
        if args.__dict__[k] is not None:
            if K not in param_dic:
                param_dic[K] = {}
            param_dic[K][k] = args.__dict__[k]
        elif yml_config is not None and K in yml_config and k in yml_config[K]:
            if K not in param_dic:
                param_dic[K] = {}
            param_dic[K][k] = yml_config[K][k]

if args.host_file is not None:
    param_dic['host_file'] = args.__dict__['host_file']

elif yml_config is not None and 'host_file' in yml_config:
    param_dic['host_file'] = yml_config['host_file']

print('PARAMETERS USED IN SIMULATION\n')
indent = '    '
for K in param_dic:
    if K == 'host_file':
        print(K + ": " + f"{param_dic['host_file']}")
        continue
    print(K + ':')
    for k in param_dic[K]:
        print(indent + f'{k}: {param_dic[K][k]}')

param_dic['yaml_path'] = args.config_path

sim = Simulator(param_dic)
sim.simulate()

if args.fit:
    sim.fit_lc()
    sim.write_fit()
