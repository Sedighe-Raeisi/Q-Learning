from src.plot import plot_check_stationary
from src.Run_from_data.Data_utils import gt_utils
import os
from src.plot import plot_check_stationary
root_path = os.getcwd()
save_dir_prefix = "CRL_chk_"
file_name = [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0]
save_path = os.path.join(root_path,file_name)

plot_check_stationary(save_path,gt_utils)