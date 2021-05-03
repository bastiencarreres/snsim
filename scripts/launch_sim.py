import argparse
import yaml
import ast
import snsim

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
        'band_dic',
        'write_format',
        'obs_config_path'],
    'db_config': [
        'dbfile_path',
        'db_cut',
        'zp',
        'gain',
        'ra_size',
        'dec_size'],
    'sn_gen': [
        'randseed',
        'n_sn',
        'sn_rate',
        'rate_pw',
        'duration',
        'nep_cut',
        'z_range',
        'v_cmb',
        'M0',
        'mag_smear',
        'smear_mod'],
    'cosmology': [
        'Om',
        'H0'],
    'salt_gen': [
        'salt_dir',
	    'version',
        'alpha',
        'beta',
        'mean_x1',
        'mean_c',
        'sig_x1',
        'sig_c'],
    'vpec_gen': [
        'host_file',
        'mean_vpec',
        'sig_vpec']}

parser.add_argument("--write_path",type=str)
parser.add_argument("--sim_name",type=str)
parser.add_argument("--band_dic",type=yaml.load)
parser.add_argument("--write_format", type=str,nargs='+')
parser.add_argument("--obs_config_path",type=str)

parser.add_argument("--dbfile_path",type=str)
parser.add_argument("--db_cut",type=yaml.load)
parser.add_argument("--zp",type=float)
parser.add_argument("--gain",type=float)
parser.add_argument("--ra_size",type=float)
parser.add_argument("--dec_size",type=float)

parser.add_argument("--randseed",type=int)
parser.add_argument("--n_sn",type=int)
parser.add_argument("--sn_rate",type=float)
parser.add_argument("--rate_pw",type=float)
parser.add_argument("--duration",type=float)

parser.add_argument("--nep_cut", action='append', nargs='+')
parser.add_argument("--z_range",type=float,nargs=2)
parser.add_argument("--v_cmb",type=float)
parser.add_argument("--M0",type=float)
parser.add_argument("--mag_smear",type=float)
parser.add_argument("--smear_mod",type=str)

parser.add_argument("--Om",type=float)
parser.add_argument("--H0",type=float)

parser.add_argument("--salt_dir",type=str)
parser.add_argument("--version",type=int)
parser.add_argument("--alpha",type=float)
parser.add_argument("--beta",type=float)
parser.add_argument("--mean_x1",type=float)
parser.add_argument("--sig_x1",type=float)
parser.add_argument("--mean_c",type=float)
parser.add_argument("--sig_c",type=float)

parser.add_argument("--host_file",type=str)
parser.add_argument("--mean_vpec",type=float)
parser.add_argument("--sig_vpec",type=float)

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

print('Parameters used :\n')
indent = '    '
for K in param_dic:
    print(K + ':')
    for k in param_dic[K]:
        print(indent + f'{k}: {param_dic[K][k]}')

param_dic['yaml_path'] = args.__dict__['config_path']

print(param_dic)
sim = snsim.sn_sim(param_dic)
sim.simulate()

if args.fit:
    sim.fit_lc()
    sim.write_fit()
