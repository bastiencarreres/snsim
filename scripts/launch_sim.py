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

################
# DATA SECTION #
################
data_grp = parser.add_argument_group('data')
data_grp.add_argument("--write_path", type=str, help="'/PATH/TO/OUTPUT'")
data_grp.add_argument("--sim_name", type=str, help="'NAME OF SIMULATION'")
data_grp.add_argument("--write_format",
                      type=str, nargs='+',
                      help="'format' or ['format1','format2']")

#########################
# SURVEY_CONFIG SECTION #
#########################
survey_grp = parser.add_argument_group('survey_config')
survey_grp.add_argument("--survey_file", type=str, help='/PATH/TO/FILE')
survey_grp.add_argument("--band_dic", type=yaml.load,
                        help="{'band1_db_key':'band1_sncosmo_key', ...} database key -> sncosmo key,\
                           put dictionnary between double quotationmark")
survey_grp.add_argument("--add_data", type=str, nargs='+',
                        help="--add_data 'key1' 'key2' ... # Add metadata from survey file to output")
survey_grp.add_argument("--survey_cut", type=yaml.load,
                        help="{'key1': ['conditon1','conditon2',...], 'key2':['conditon1'],...},\
                          put dictionnary between double quotation mark")
survey_grp.add_argument("--zp", type=float, help="INSTRUMENTAL ZEROPOINT")
survey_grp.add_argument("--sig_zp", type=float, help="UNCERTAINTY ON ZEROPOINT")
survey_grp.add_argument("--sig_psf", type=float, help="GAUSSIAN PSF SIGMA")
survey_grp.add_argument("--noise_key", type=str, nargs=2,
                        help="--noise_key key type, type can be 'mlim5' or 'skysigADU'")
survey_grp.add_argument("--gain", type=float, help="CCD GAIN e-/ADU")
survey_grp.add_argument("--ra_size", type=float, help="RA FIELD SIZE in DEG")
survey_grp.add_argument("--dec_size", type=float, help="Dec FIELD SIZE in DEG")
survey_grp.add_argument("--start_day", type=date_read,
                        help="Survey starting day MJD NUMBER or 'YYYY-MM-DD'")
survey_grp.add_argument("--end_day", type=date_read,
                        help="Survey ending day MJD NUMBER or 'YYYY-MM-DD'")
survey_grp.add_argument("--duration", type=float, help="SURVEY DURATION IN DAYS")
survey_grp.add_argument("--sub_field", type=str, help="SUBFIELD KEY")
survey_grp.add_argument("--field_map", type=str, help="SUBFIELD MAP")
survey_grp.add_argument("--fake_skynoise", nargs=2, help="[VALUE, 'add' or 'replace']")
survey_grp.add_argument("--key_dic", type=yaml.load,
                        help="Change column(s) name to correspond to what is needed")

###################
# SN_GEN SECTION #
##################
sngen_grp = parser.add_argument_group('sn_gen')
sngen_grp.add_argument("--randseed", type=int, help="RANDSEED TO REPRODUCE SIMULATION")
sngen_grp.add_argument("--duration_for_rate", type=float,
                       help="FAKE DURATION ONLY USED TO GENERATE N SN (DAYS)")
sngen_grp.add_argument("--n_sn", type=int, help="NUMBER OF SN TO GENERATE")
sngen_grp.add_argument("--sn_rate", help="rate of SN/Mpc^3/year or 'ptf19'")
sngen_grp.add_argument("--rate_pw", type=float, help="rate = sn_rate*(1+z)^rate_pw")
sngen_grp.add_argument("--nep_cut", action='append', nargs='+',
                       help="--nep_cut nep_min1 Tmin Tmax --nep_cut nep_min2 Tmin2 Tmax2 'filter1',\
                    put cuts on the number of epochs between Tmin and Tmax \
                    (restframe, relative to peak), optionaly in a selected filter")
sngen_grp.add_argument("--z_range", type=float, nargs=2,
                       help="--zrange zmin zmax, Cosmological redshift range")
sngen_grp.add_argument("--M0", help="SN ABSOLUT MAGNITUDE")
sngen_grp.add_argument("--mag_sct", type=float, help="SN INTRINSIC COHERENT SCATTERING")
sngen_grp.add_argument("--sct_mod", type=str,
                       help="'G10','C11_i' USE WAVELENGHT DEP MODEL FOR SN INT SCATTERING")

#####################
# COSMOLOGY SECTION #
#####################
cosmology_grp = parser.add_argument_group('cosmology')
cosmology_grp.add_argument("--name", type=str, help="ASTROPY COSMOLOGICAL MODEL TO LOAD")
cosmology_grp.add_argument("--Om0", type=float, help="MATTER DENSITY")
cosmology_grp.add_argument("--H0", type=float, help="HUBBLE CONSTANT")

###############
# CMB SECTION #
###############
cmb_grp = parser.add_argument_group('cmb')
cmb_grp.add_argument("--v_cmb", type=float, help="OUR PECULIAR VELOCITY")
cmb_grp.add_argument("--ra_cmb", type=float, help="GAL L OF CMB DIPOLE")
cmb_grp.add_argument("--dec_cmb", type=float, help="GAL B OF CMB DIPOLE")

########################
# MODEL_CONFIG SECTION #
########################
modelcfg_grp = parser.add_argument_group('model_config')
modelcfg_grp.add_argument("--model_dir", type=str, help="'/PATH/TO/SALT/MODEL'")
modelcfg_grp.add_argument("--model_name", type=int, help="'THE MODEL NAME', example 'salt2'")
modelcfg_grp.add_argument("--mw_dust", type=str, nargs='+',
                          help="--mw_dust 'MODEL_NAME' or --mw_dust ['MODEL_NAME', RV]")

modelcfg_grp.add_argument("--mod_fcov", type=bool, help="APPLY MODEL COV TO FLUX True or False")
# SALT PARAM
modelcfg_grp.add_argument("--alpha", type=float, help="STRETCH CORRECTION")
modelcfg_grp.add_argument("--beta", type=float, help="COLOR CORRECTION")
modelcfg_grp.add_argument("--dist_x1", nargs='+', help="MEAN SIGMA; MEAN SIGMA- SIGMA+ or 'N21'")
modelcfg_grp.add_argument("--dist_c", type=float, nargs='+',
                          help="MEAN SIGMA or MEAN SIGMA- SIGMA+")

#####################
# VPEC_DIST SECTION #
#####################
vpecdist_grp = parser.add_argument_group('vpec_dist')
vpecdist_grp.add_argument("--mean_vpec", type=float, help="MEAN SN PECULIAR VELOCITY")
vpecdist_grp.add_argument("--sig_vpec", type=float, help="SIGMA PECULIAR VELOCITY")

#####################
# HOST SECTION #
#####################
host_grp = parser.add_argument_group('host')
host_grp.add_argument("--host_file", type=str, help="'/PATH/TO/HOSTFILE'")
host_grp.add_argument("--distrib", type=str, help="'as_sn', 'as_host' or 'mass_weight'")
host_grp.add_argument("--key_dic", type=yaml.load,
                      help="Change column(s) name to correspond to what is needed")

########################
# ALPHA_DIPOLE SECTION #
########################
alphad_grp = parser.add_argument_group('alpha_dipole')
alphad_grp.add_argument("--alpha_coord", dest="coord", type=float, nargs=2,
                        help="--alpha_coord RA Dec, Alpha dipole coordinates")
alphad_grp.add_argument("--alpha_A", dest="A", type=float,
                        help="Alpha dipole = A + B * cos(theta)")
alphad_grp.add_argument("--alpha_B", dest="B", type=float,
                        help="Alpha dipole = A + B * cos(theta)")

args = parser.parse_args()

args_groups = {}
for group in parser._action_groups:
    group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
    args_groups[group.title] = argparse.Namespace(**group_dict)

if args_groups['sn_gen'].nep_cut is not None:
    args.nep_cut = nep_cut(args.nep_cut)

if args.mw_dust is not None:
    args.mw_dust = mw_dust_read(args.mw_dust)

if args_groups['model_config'].dist_x1 is not None:
    if args.dist_x1 not in ['N21']:
        args.dist_x1 = list(args.dist_x1)
        for i in range(len(args.dist_x1)):
            args.dist_x1[i] = float(args.dist_x1[i])

with open(args.config_path, "r") as f:
    yml_config = yaml.safe_load(f)

param_dic = {}
for K in args_groups:
    param_dic[K] = {}

for K in args_groups:
    for k in args_groups[K].__dict__:
        if args_groups[K].__dict__[k] is not None:
            param_dic[K][k] = args_groups[K].__dict__[k]
        elif yml_config is not None and K in yml_config and k in yml_config[K]:
            param_dic[K][k] = yml_config[K][k]

print('PARAMETERS USED IN SIMULATION\n')
indent = '    '
for K in param_dic:
    if param_dic[K] == {}:
        param_dic[K] = None
        continue
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
    sim.sn_sample.fit_lc()
    sim.sn_sample.write_fit()
