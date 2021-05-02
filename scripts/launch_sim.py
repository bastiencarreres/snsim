import argparse
import yaml
import ast
import snsim

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

ignore_keys = ['config_path', 'fit']

keys_list = []
for K in keys_dic:
    keys_list += keys_dic[K]

for k in keys_list:
    parser.add_argument(f"--{k}")

args = parser.parse_args()

for arg in args.__dict__:
    if arg not in keys_list and arg not in ignore_keys:
        raise ValueError(f"{arg} option doesn't exist")
print(args)
param_dic = {}

with open(args.config_path, "r") as f:
    yml_config = yaml.safe_load(f)

for K in keys_dic:
    if K not in yml_config:
        param_dic[K]={}
    for k in keys_dic[K]:
        if args.__dict__[k] is not None:
            param_dic[K][k] = args
        elif K in yml_config:
            if K not in param_dic:
                param_dic[K] = {}
            if k in yml_config[K]:
                param_dic[K][k] = yml_config[K][k]

print('Parameters used :\n')
indent = '    '
for K in param_dic:
    print(K + ':')
    for k in param_dic[K]:
        print(indent + f'{k}: {param_dic[K][k]}')

param_dic['yaml_path'] = args.__dict__['config_path']

print(param_dic)
#sim = snsim.sn_sim(param_dic)
#sim.simulate()

if args.fit:
    sim.fit_lc()
    sim.write_fit()
