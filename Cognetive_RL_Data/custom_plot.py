import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=64"
from src.custom_plot import Custom_plot
from src.Run_from_data.Data_utils import gt_utils,realparame2gtarray, generate_pdf

save_dir_prefix = "CRL_chk_"

to_plot = [[0,1],[0,2]]

plot_dict = {"est_color": "blue", "gt_color": "red", "flat_color":"green","pdf_color":"grey",
             "legend": None, "xlabel_fontsize": 23, "title_fontsize": None,
             "max_y_limit": 40.3,"save_name":"custom_with_pdf","pdf_fill":False}

Custom_plot(generate_pdf, pdf_state=True, ground_truth = True, HB_Est = True, FlatB_Est = True,TABLE = False,
            gt_utils=gt_utils, realparame2gtarray=realparame2gtarray, save_dir_prefix=save_dir_prefix,
           fighigth=3, figwidth=12, n_rows=1, n_cols=2,
           scaler=None, to_plot=to_plot, plot_dict=plot_dict)

