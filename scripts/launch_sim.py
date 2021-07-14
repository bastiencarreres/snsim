import argparse
import yaml
from snsim import Simulator

def nep_cut(nepc):
    for i in range(len(nepc)):
        if len(nepc[i])>=1:
            nepc[i][0]=int(nepc[i][0])
        if len(nepc[i])>=3:
            nepc[i][1]=float(nepc[i][1])
            nepc[i][2]=float(nepc[i][2])
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
        'duration'],
    'sn_gen': [
        'randseed',
        'duration_for_rate',
        'n_sn',
        'sn_rate',
        'rate_pw',
        'nep_cut',
        'z_range',
        'M0',
        'mag_smear',
        'smear_mod'],
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
        'sig_vpec']}

parser.add_argument("--write_path",type=str)
parser.add_argument("--sim_name",type=str)
parser.add_argument("--band_dic",type=yaml.load)
parser.add_argument("--write_format", type=str, nargs='+')

parser.add_argument("--survey_file",type=str)
parser.add_argument("--add_data", type=str, nargs='+')
parser.add_argument("--db_cut",type=yaml.load)
parser.add_argument("--zp",type=float)
parser.add_argument("--sig_zp",type=float)
parser.add_argument("--sig_psf",type=float)
parser.add_argument("--noise_key", type=str, nargs=2)
parser.add_argument("--gain",type=float)
parser.add_argument("--ra_size",type=float)
parser.add_argument("--dec_size",type=float)
parser.add_argument("--start_day",type=float)
parser.add_argument("--end_day",type=float)
parser.add_argument("--duration",type=float)

parser.add_argument("--randseed",type=int)
parser.add_argument("--duration_for_rate",type=float)
parser.add_argument("--n_sn",type=int)
parser.add_argument("--sn_rate",type=float)
parser.add_argument("--rate_pw",type=float)
parser.add_argument("--nep_cut", action='append', nargs='+')
parser.add_argument("--z_range", type=float, nargs=2)
parser.add_argument("--M0",type=float)
parser.add_argument("--mag_smear",type=float)
parser.add_argument("--smear_mod",type=str)

parser.add_argument("--Om0",type=float)
parser.add_argument("--H0",type=float)

parser.add_argument("--v_cmb",type=float)
parser.add_argument("--ra_cmb",type=float)
parser.add_argument("--dec_cmb",type=float)

parser.add_argument("--model_dir",type=str)
parser.add_argument("--model_name",type=int)
parser.add_argument("--alpha",type=float)
parser.add_argument("--beta",type=float)
parser.add_argument("--mean_x1",type=float)
parser.add_argument("--sig_x1",type=float)
parser.add_argument("--mean_c",type=float)
parser.add_argument("--sig_c",type=float)

parser.add_argument("--mean_vpec",type=float)
parser.add_argument("--sig_vpec",type=float)

parser.add_argument("--host_file",type=str)

args = parser.parse_args()

if args.nep_cut is not None:
    args.nep_cut = nep_cut(args.nep_cut)

param_dic = {}

with open(args.config_path, "r") as f:
    yml_config = yaml.safe_load(f)


for K in keys_dic:
    param_dic[K]={}
    for k in keys_dic[K]:
        if args.__dict__[k] is not None:
            param_dic[K][k] = args.__dict__[k]
        elif k in yml_config[K]:
            param_dic[K][k] = yml_config[K][k]

if args.__dict__['host_file'] is not None:
    param_dic['host_file'] = args.__dict__['host_file']

elif 'host_file' in yml_config:
        param_dic['host_file'] = yml_config['host_file']

print('Parameters used :\n')
indent = '    '
for K in param_dic:
    print(K + ':')
    for k in param_dic[K]:
        print(indent + f'{k}: {param_dic[K][k]}')

param_dic['yaml_path'] = args.__dict__['config_path']

print(param_dic)
sim = Simulator(param_dic)
sim.simulate()

if args.fit:
    sim.fit_lc()
    sim.write_fit()
