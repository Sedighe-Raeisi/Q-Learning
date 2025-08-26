import os.path
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"

from src.model import Flat_HSModel
from src.flat_mcmc_utils import run_mcmc
from src.Run_from_data.Data_utils import mix_data,gt_utils,realparame2gtarray, generate_pdf
from src.flat_plot import plt_mcmc
print("---------------------- parameter defining ------------------------")

N_param_set = 100
NUM_WARMUP = 1000
NUM_CHAINS = 3
NUM_SAMPLES = 1000
NUM_BATCH_SAMPLES = 5
root_path = os.getcwd()
save_dir_prefix = "CRL_chk_"

# Define parameters for RLC circuit
Alpha_info = { "Alpha_mean": .3, "Alpha_std": 0.02}
ForgetRate_info = {"ForgetRate_mean": .4, "ForgetRate_std":0.01}
Session_info = {"n_trials_per_session":100, "n_sessions":2}

model = Flat_HSModel

system_param_dict = {"N_param_set":N_param_set,"Alpha_info":Alpha_info, "ForgetRate_info":ForgetRate_info, 'Session_info': Session_info} # Updated parameter dictionary
mode = "plot" #"plot" or "run"
print(f"--------------------------- mode = {mode} --------------------------------")
if mode == "run":
    print("----------------------- run mcmc_utils -----------------------")
    run_mcmc(system_param_dict ,
                 NUM_WARMUP=NUM_WARMUP, NUM_CHAINS=NUM_CHAINS, NUM_SAMPLES=NUM_SAMPLES, NUM_BATCH_SAMPLES=NUM_BATCH_SAMPLES,
                 root_path = root_path, save_dir_prefix = save_dir_prefix,
                 program_state = "start", model = model,
                 display_svi = True, mix_data = mix_data, gt_utils = gt_utils)
elif mode=="plot":

    true_params_file_str = f"chk_GT_Data.pkl"
    gt_save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])
    est_save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith('Flat_'+save_dir_prefix)][0])

    plt_mcmc(gt_save_path,est_save_path, gt_utils, realparame2gtarray, generate_pdf, true_params_file_str,
                 stop_subplot_n=None, figlength=3,
                 complex_pdf=True, x_range=None, scaler=None)