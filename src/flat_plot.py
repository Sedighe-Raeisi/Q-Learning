import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib as mpl
import numpyro


def plt_mcmc(gt_save_path,est_save_path, gt_utils,realparame2gtarray, generate_pdf, true_params_file_str,
             stop_subplot_n=None, figlength = 3,
             complex_pdf = False, x_range = None, scaler=None):
    """

    :param save_path:
    :param gt_utils:
    :param realparame2gtarray: converts a dic of gt to array of gt
    :param generate_pdf: the function that is used to generate the arrays of pdf distributions
    :param true_params_file_str: the name of the file that contains true values or ground truth values
    :param stop_subplot_n: if we want to have a few number of relevent sub-plot in figure
    :param width_scale: scale of the plot for width which is used in fig_size
    :param hight_scale: scale of the plot for hight which is used in fig_size
    :param complex_pdf: if we have coef which is computed from some basic coef of the system like k/m
    :param x_range: list of the form [x_min,y_max]
    :return:
    """
    mpl.rcParams['xtick.labelsize'] = 6
    true_params_filename = os.path.join(gt_save_path, true_params_file_str)
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)

    gt_coef_array = realparame2gtarray(true_params)
    gt_dict = gt_utils(true_params)
    eqs = gt_dict['eqs']
    coef_names = gt_dict['coef_names']
    gt_coef = gt_dict['gt_coef']
################## preparing est distribution ######################
    if scaler is None:

        npz_sample_path = os.path.join(est_save_path, "mcmc_samples.npz")
        loaded_samples = np.load(npz_sample_path, allow_pickle=True)
        est_coef = loaded_samples['coef']
        est_coef_array = np.array(est_coef)
    else:
        with open(os.path.join(est_save_path, f'revert_mcmc_samples.pkl'), 'rb') as f:
            mcmc_coef_results = pickle.load(f)
            est_coef_array = np.array(mcmc_coef_results)
################### preparing pdf samples #########################
    if complex_pdf == False:
        gt_info_arr = gt_dict['gt_info_arr']
        pdf_samples = generate_pdf(gt_info_arr)
    # else:
    #     pdf_samples = generate_pdf(gt_save_path)
################### parameters of plot defining  #########################
    N_Eqs = est_coef_array.shape[2]
    N_Coef = est_coef_array.shape[1]
    est_coef_mean_arr = np.mean(est_coef_array, axis=0)
    n_plot_cols = stop_subplot_n if stop_subplot_n else N_Coef
    fig, ax = plt.subplots(n_plot_cols, N_Eqs, figsize=(figlength * N_Eqs, figlength * n_plot_cols))

    fig.subplots_adjust(wspace=0)
    stationary_sample_i = 0  # 3000 #this option made no difference
    for EQ_ID in range(N_Eqs):
        # fig.text(0.5*(1/N_Eqs)+EQ_ID*(1/N_Eqs), 1, eqs[EQ_ID].split("=")[0],color = 'purple')

        for coef_i in range(n_plot_cols):
            if N_Eqs == 1:
                axi = ax[coef_i]
            else:
                axi = ax[coef_i, EQ_ID]
            est_data_values = np.squeeze(est_coef_array[stationary_sample_i:, coef_i, EQ_ID].reshape(-1))
            if true_params:
                gt_data_values = np.squeeze(gt_coef_array[EQ_ID, coef_i, :])
                gt_data = pd.DataFrame({
                    'value': gt_data_values,
                    'coef': coef_names[coef_i],
                    'type': 'gt'
                })

            est_data = pd.DataFrame({
                'value': est_data_values,
                'coef': coef_names[coef_i],
                'type': 'est'
            })




            try:
                sns.histplot(est_data_values, bins=50, kde=True, ax=axi, color='blue',
                             alpha=0.3, stat="density")
                if true_params:
                    sns.histplot(gt_coef_array[EQ_ID, coef_i, :], bins=50, kde=True, ax=axi, color='green',
                                 alpha=0.3, stat="density")
                # ax[coef_i,EQ_ID].hist(pdf_samples[EQ_ID, i, :], bins=50, density=True, alpha=0.4 , color='red')#kde=True, ax=axes[EQ_ID,i], stat="density")
            except:
                sns.histplot(est_data_values, kde=True, ax=axi, color='blue',
                             alpha=0.3,
                             stat="density")
                if true_params:
                    sns.histplot(gt_coef_array[EQ_ID, coef_i, :], kde=True, ax=axi, color='green', alpha=0.3,
                                 stat="density")

            max_y_limit = .98  # A good starting value, adjust as needed
            axi.set_ylim(0, max_y_limit)
            # if axi.get_ylim()[1] > max_y_limit:
            #     axi.set_ylim(0, max_y_limit)

            est_mean = round(np.mean(est_coef_mean_arr[coef_i, EQ_ID]), 2)
            est_std = round(np.std(est_coef_mean_arr[coef_i, EQ_ID]), 2)
            if true_params:
                gt_mean = round(np.mean(gt_coef_array[EQ_ID, coef_i, :]), 4)
                gt_std = round(np.std(gt_coef_array[EQ_ID, coef_i, :]), 4)

            axi.axvline(est_mean, color='blue', linestyle='--',
                                      label=f'est_mean = "%.3f", est_std = "%.3f"' % (
                                      est_mean, est_std))  # ,Est std={est_std}')
            if true_params:
                axi.axvline(gt_mean, color='green', linestyle='--',
                                          label='gt_mean = "%.3f", gt_std = "%.3f" ' % (gt_mean, gt_std))

            axi.axhline(0, color='black', linestyle='-')

            if true_params is not None and gt_std == 0.0:
                axi.set_xlim(-2, 2)

            if EQ_ID == 0:
                axi.set_ylabel(coef_names[coef_i], fontsize=12)
            else:
                axi.set_ylabel("")
            if coef_i == 0:
                axi.set_title(f'{eqs[EQ_ID]}', fontsize=10, color='purple')
            else:
                axi.set_title("")

            axi.xaxis.set_major_locator(plt.MaxNLocator(3))
            axi.set_yticks(np.array([0]))
            axi.set_yticklabels([""], fontsize=12, rotation=90)
            axi.yaxis.set_tick_params(length=0)
            # .xticks(fontsize=12)
            axi.spines['left'].set_visible(False)
            axi.spines['top'].set_visible(False)
            axi.spines['right'].set_visible(False)

            axi.legend(
                loc='upper center', bbox_to_anchor=(0.5, 1.03),

                ncol=1, fancybox=True, shadow=True, fontsize=8)


    plt.tight_layout()
    plt.savefig(os.path.join(est_save_path,'dist_plot.jpg'))
    plt.show()
