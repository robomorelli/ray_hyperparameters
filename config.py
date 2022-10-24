import os
from ray import tune

root = os.getcwd()
model_results_path = root + '/model_results/'

if 'fdir' in root:
    print('paths should be fine')
else:
    root = os.path.join(root,'artificial_intelligence/repos/fdir')

config_path = root + '/train_configurations/'

ray_mapper = {'tune.choice': tune.choice}

vae_config_file = 'vae.yaml'
ae_config_file = 'ae.yaml'
cnn3d_config_file = 'cnn3d.yaml'

# nls_kdd use case
nls_kdd_datapath = root + '/data/nls_kdd/'
nls_kdd_cols = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", 'difficulty']

nls_kdd_cat_cols = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
cont_cols = [x for x in nls_kdd_cols if x not in nls_kdd_cat_cols]