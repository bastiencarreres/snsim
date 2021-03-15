import argparse
import yaml
import ast
import snsim

parser = argparse.ArgumentParser()
parser.add_argument("config_path", default=None, type=str,help="Configuration file")
parser.add_argument("-fit", action='store_true')

keys_dic = {'data': ['write_path','sim_name','band_dic','obs_config_path'],
            'db_config':['dbfile_path','zp','gain'],
            'sn_gen':['randseed','n_sn','nep_cut','z_range','v_cmb','M0','mag_smear'],
            'cosmology':['Om','H0'],
            'salt2_gen':['salt2_dir','alpha','beta','mean_x1','mean_c','sig_x1','sig_c'],
            'vpec_gen':['host_file','mean_vpec','sig_vpec']
            }
type_dic={'write_path':str,
          'sim_name':str,
          'band_dic':dict,
          'obs_cfg_path':str,
          'dbfile_path':str,
          'zp':float,
          'gain':float,
          'randseed':int,
          'n_sn':int,
          'z_range':list,
          'v_cmb':float,
          'M0':float,
          'mag_smear':float,
          'Om':float,
          'H0':float,
          'salt2_dir':str,
          'alpha':float,
          'beta':float,
          'mean_x1':float,
          'mean_c':float,
          'sig_x1':float,
          'sig_c':float,
          'host_file':str,
          'mean_vpec':float,
          'sig_vpec':float,
          'mean_x1':float}

ignore_keys=['config_path','fit']

keys_list = []
for K in keys_dic : keys_list += keys_dic[K]

for k in keys_list:
    parser.add_argument(f"--{k}")

args = parser.parse_args()

for arg in args.__dict__:
    if arg not in keys_list and arg not in ignore_keys:
        print(f"{arg} option doesn't exist, arg ignored")

param_dic={}

with open(args.config_path, "r") as f:
    yml_config = yaml.safe_load(f)

for K in keys_dic:
    for k in keys_dic[K]:
        if args.__dict__[k] is not None:
            if K not in param_dic:
                param_dic[K]={}
            if k == 'nep_cut':
                try :
                    int(k)
                    type_dic[k]=int
                except:
                    type_dic[k]=list
            if type_dic[k]==list:
                param_dic[K][k]=ast.literal_eval(args.__dict__[k])
            else:
                param_dic[K][k]=type_dic[k](args.__dict__[k])
        elif K in yml_config:
            if K not in param_dic:
                param_dic[K]={}
            if k in yml_config[K]:
                param_dic[K][k]=yml_config[K][k]

print('Parameters used :\n')
indent='    '
for K in param_dic:
    print(K+':')
    for k in param_dic[K]:
        print(indent+f'{k}: {param_dic[K][k]}')

param_dic['yaml_path'] = args.__dict__['config_path']

sim = snsim.sn_sim(param_dic)
sim.simulate()

if args.fit:
    sim.fit_lc()
