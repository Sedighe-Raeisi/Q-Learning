import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib as mpl
import numpyro

def plt_mcmc(save_path,gt_utils,realparame2gtarray, generate_pdf, true_params_file_str,
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
    true_params_filename = os.path.join(save_path, true_params_file_str)
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)

    gt_coef_array = realparame2gtarray(true_params)
    gt_dict = gt_utils(true_params)
    eqs = gt_dict['eqs']
    coef_names = gt_dict['coef_names']
    gt_coef = gt_dict['gt_coef']
################## preparing est distribution ######################
    if scaler is None:

        npz_sample_path = os.path.join(save_path, "mcmc_samples.npz")
        loaded_samples = np.load(npz_sample_path, allow_pickle=True)
        est_coef = loaded_samples['coef']
        est_coef_array = np.array(est_coef)
    else:
        with open(os.path.join(save_path, f'revert_mcmc_samples.pkl'), 'rb') as f:
            mcmc_coef_results = pickle.load(f)
            est_coef_array = np.array(mcmc_coef_results)
################### preparing pdf samples #########################
    if complex_pdf == False:
        gt_info_arr = gt_dict['gt_info_arr']
        pdf_samples = generate_pdf(gt_info_arr)
    else:
        pdf_samples = generate_pdf(save_path)
################### parameters of plot defining  #########################
    N_Eqs = est_coef_array.shape[3]
    N_Coef = est_coef_array.shape[2]
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
            est_data_values = np.squeeze(est_coef_array[stationary_sample_i:, :, coef_i, EQ_ID].reshape(-1))
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
                sns.histplot(est_coef_mean_arr[:, coef_i, EQ_ID], bins=50, kde=True, ax=axi, color='blue',
                             alpha=0.3, stat="density")
                if true_params:
                    sns.histplot(gt_coef_array[EQ_ID, coef_i, :], bins=50, kde=True, ax=axi, color='green',
                                 alpha=0.3, stat="density")
                # ax[coef_i,EQ_ID].hist(pdf_samples[EQ_ID, i, :], bins=50, density=True, alpha=0.4 , color='red')#kde=True, ax=axes[EQ_ID,i], stat="density")
            except:
                sns.histplot(est_coef_mean_arr[:, coef_i, EQ_ID], kde=True, ax=axi, color='blue',
                             alpha=0.3,
                             stat="density")
                if true_params:
                    sns.histplot(gt_coef_array[EQ_ID, coef_i, :], kde=True, ax=axi, color='green', alpha=0.3,
                                 stat="density")
            est_mean = round(np.mean(est_coef_mean_arr[:, coef_i, EQ_ID]), 2)
            est_std = round(np.std(est_coef_mean_arr[:, coef_i, EQ_ID]), 2)
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
    plt.savefig(os.path.join(save_path,'dist_plot.jpg'))
    plt.show()

def plot_scatter_gt_xy(save_path,gt_utils,realparame2gtarray, true_params_file_str, stop_subplot_n = None,width_scale = 4, hight_scale = 4):
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
    true_params_filename = os.path.join(save_path, true_params_file_str)
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)
    gt_coef_array = realparame2gtarray(true_params)
    gt_dict = gt_utils(true_params)
    eqs = gt_dict['eqs']
    coef_names = gt_dict['coef_names']
    gt_coef = gt_dict['gt_coef']
    # Shape of X_data: (n_indv, n_coef, n_obs)
    # Shape of Y_data: (n_indv, n_eq, n_obs)
    N_Eqs = Y_data.shape[1]
    N_Coef = X_data.shape[1]
    n_plot_cols = stop_subplot_n if stop_subplot_n else N_Coef
    fig, axes = plt.subplots(N_Eqs, n_plot_cols,
                             figsize=(width_scale * n_plot_cols, hight_scale * N_Eqs))  # 2 rows, 3 columns

    for eq_i in range(N_Eqs):

        for coef_i in range(n_plot_cols):  # Loop over coefficients
            if N_Eqs == 1:
                axi = axes[coef_i]
            else:
                axi = axes[coef_i, eq_i]
            print(f'sub plot {coef_i} is created')
            axi.scatter(X_data[:,coef_i,:],Y_data[:,eq_i,:], alpha=0.5, s=10)
# Add titles and labels for clarity
            axi.set_title(f'Eq : {eqs[eq_i]}')
            axi.set_xlabel(f'Coef: {coef_names[coef_i]} ')
            axi.set_ylabel(f'{eqs[eq_i].split("=")[0]}')

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,"traj_scatter_plot.jpg"))
    plt.show() # Display the plot

def plot_all_traj(save_path,gt_utils,realparame2gtarray, true_params_file_str, stop_subplot_n = None,width_scale = 4, hight_scale = 4):
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
    true_params_filename = os.path.join(save_path, true_params_file_str)
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)
    gt_coef_array = realparame2gtarray(true_params)
    gt_dict = gt_utils(true_params)
    eqs = gt_dict['eqs']
    coef_names = gt_dict['coef_names']
    gt_coef = gt_dict['gt_coef']
    # Shape of X_data: (n_indv, n_coef, n_obs)
    # Shape of Y_data: (n_indv, n_eq, n_obs)
    N_Eqs = Y_data.shape[1]
    N_Coef = X_data.shape[1]
    N_indv = X_data.shape[0]
    n_plot_cols = stop_subplot_n if stop_subplot_n else N_Coef
    fig, axes = plt.subplots(N_Eqs, n_plot_cols,
                             figsize=(width_scale * n_plot_cols, hight_scale * N_Eqs))  # 2 rows, 3 columns

    for eq_i in range(N_Eqs):

        for coef_i in range(n_plot_cols):  # Loop over coefficients
            if N_Eqs == 1:
                axi = axes[coef_i]
            else:
                axi = axes[coef_i, eq_i]
            print(f'sub plot {coef_i} is created')
            for traj_i in range(N_indv):
                axi.scatter(X_data[traj_i,coef_i,:],Y_data[traj_i,eq_i,:], alpha=0.5, s=1)
# Add titles and labels for clarity
            axi.set_title(f'Eq : {eqs[eq_i]}')
            axi.set_xlabel(f'Coef: {coef_names[coef_i]} ')
            axi.set_ylabel(f'{eqs[eq_i].split("=")[0]}')

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,"all_traj_plot.jpg"))
    plt.show()

def plot_std(save_path,gt_utils,realparame2gtarray, true_params_file_str, stop_subplot_n = None,width_scale = 4, hight_scale = 4):
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
    true_params_filename = os.path.join(save_path, true_params_file_str)
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)
    gt_coef_array = realparame2gtarray(true_params)
    gt_dict = gt_utils(true_params)
    eqs = gt_dict['eqs']
    coef_names = gt_dict['coef_names']
    gt_coef = gt_dict['gt_coef']
    # Shape of X_data: (n_indv, n_coef, n_obs)
    # Shape of Y_data: (n_indv, n_eq, n_obs)
    N_Eqs = Y_data.shape[1]
    N_Coef = X_data.shape[1]
    N_indv = X_data.shape[0]



    xstd_s = np.std(X_data,0)
    ystd_s = np.std(Y_data,0)
    n_plot_cols = stop_subplot_n if stop_subplot_n else N_Coef
    fig, axes = plt.subplots(n_plot_cols,N_Eqs,
                             figsize=(width_scale * n_plot_cols, hight_scale * N_Eqs))  # 2 rows, 3 columns
    instead_t = np.arange(0,xstd_s.shape[1])
    for eq_i in range(N_Eqs):

        for coef_i in range(n_plot_cols):  # Loop over coefficients
            print(f'sub plot {coef_i} is created')
            if N_Eqs == 1:
                axi = axes[coef_i]
            else:
                axi = axes[coef_i, eq_i]
            xtarget = xstd_s[coef_i,:]
            ytarget = ystd_s[eq_i,:]
            axi.scatter(instead_t, xtarget, alpha=0.5, s=1,label = f"std of {coef_names[coef_i]}")
            axi.scatter(instead_t, ytarget, alpha=0.5, s=1, label=f"std of {eqs[eq_i].split('=')[0]}")

# Add titles and labels for clarity
            axi.set_title(f'Coef: {coef_names[coef_i]}')
            axi.set_xlabel(f'Time')
            axi.set_ylabel(f'std')
            axi.legend (loc="upper right")

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,"std_plot.jpg"))
    plt.show()


############################################ Plot model #######################
def plot_model(model, name):

    X = np.random.rand(10,2,4)
    Y = np.random.rand(10,1,4)
    plt.figure(figsize=(5, 5),dpi=100)
    graph = numpyro.render_model(model, model_args=(X, Y))
    graph.render(name, format="png")
    plt.imshow(plt.imread(os.path.join(os.getcwd(),f"{name}.png")))
    plt.axis('off')
    plt.show()

######################################### Plot_check if MCMC got stationary ########################
def plot_check_stationary(chk_path,gt_utils):
    with open(os.path.join(chk_path, f'stationary_mcmc_samples.pkl'), 'rb') as f:
        stationary_mcmc_samples = pickle.load(f)
    # print(stationary_mcmc_samples.shape)
    n_chains , n_samples, n_indv, n_coef, n_eqs = stationary_mcmc_samples.shape
    true_params_filename = os.path.join(chk_path, "chk_GT_Data.pkl")
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)

    gt_dict = gt_utils(true_params)
    eqs = gt_dict['eqs']
    coef_names = gt_dict['coef_names']

    scale_factor = 4
    fig, ax = plt.subplots(n_coef, n_eqs, figsize=(scale_factor * n_eqs, scale_factor * n_coef))
    for eq_i in range(n_eqs):
        for coef_i in range(n_coef):
            # fig.text(-0.12, 0.1+(1/N_Coef)*coef_i, coef_names[coef_i])

            if n_eqs == 1:
                axi = ax[coef_i]
            else:
                axi = ax[coef_i, eq_i]
            for indv_i in range(n_indv):
                for chain_i in range(n_chains):
                    target = stationary_mcmc_samples[chain_i,:, indv_i, coef_i, eq_i]
                    # print(target.shape)
                    axi.plot(target)

                    if eq_i == 0:
                        axi.set_ylabel(coef_names[coef_i], fontsize=12)
                    else:
                        axi.set_ylabel("")
                    if coef_i == 0:
                        axi.set_title(f'{eqs[eq_i]}', fontsize=10, color='purple')
                    else:
                        axi.set_title("")

    plt.tight_layout()
    plt.savefig(os.path.join(chk_path, 'stationary_plot.jpg'))
    plt.show()

################## row plot ######################

import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib import gridspec

import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib import gridspec


def row_result(save_path, gt_utils, realparame2gtarray, true_params_file_str,
               fighigth=3, figwidth=12, n_rows=None, n_cols=None,
               scaler=None, to_plot=None, plot_dict=dict()):
    legend_True = True if plot_dict is not None and plot_dict.get('legend') else None
    xlabel_fontsize = plot_dict.get('xlabel_fontsize', None)
    title_fontsize = plot_dict.get('title_fontsize', None)
    est_color = plot_dict.get('est_color', 'blue')
    gt_color = plot_dict.get('gt_color', 'pink')

    ############################## Load GT Data ###########################
    true_params_filename = os.path.join(save_path, true_params_file_str)
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)

    gt_coef_array = realparame2gtarray(true_params)
    gt_dict = gt_utils(true_params)
    eqs = gt_dict['eqs']
    coef_names = gt_dict['coef_names']

    ########################## Load EST data #############################
    if scaler is None:
        npz_sample_path = os.path.join(save_path, "mcmc_samples.npz")
        loaded_samples = np.load(npz_sample_path, allow_pickle=True)
        est_coef = loaded_samples['coef']
        est_coef_array = np.array(est_coef)
    else:
        with open(os.path.join(save_path, f'revert_mcmc_samples.pkl'), 'rb') as f:
            print(f" start loading samples from {os.path.join(save_path, f'revert_mcmc_samples.pkl')}")
            mcmc_coef_results = pickle.load(f)
            est_coef_array = np.array(mcmc_coef_results)

    ################### Prepare plot data and table data ##################
    print(f"est_coef_array.shape = {est_coef_array.shape}")
    N_Eqs = est_coef_array.shape[3]
    N_Coef = est_coef_array.shape[2]
    est_coef_mean_arr = np.mean(est_coef_array, axis=0)

    mpl.rcParams['xtick.labelsize'] = 6

    # KEY CHANGE: Create a GridSpec with 2*n_rows for titles and plots

    # gs = gridspec.GridSpec(2 * n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.5],
    #                        height_ratios=[0.1] + [1] + ([0.1, 1] * (n_rows - 1)))
    # fig = plt.figure(figsize=(figwidth, fighigth * n_rows))

    missed_coef_data = []

    # Identify missed coefficients and calculate their values for the table
    for eq_i in range(N_Eqs):
        for coef_i in range(N_Coef):
            # print(f"plot index = {[eq_i, coef_i]}")
            if [eq_i, coef_i] not in to_plot:
                print(f"not in plot index = {[eq_i, coef_i]}")
                est_mean = np.mean(est_coef_mean_arr[:, coef_i, eq_i])
                est_std = np.std(est_coef_mean_arr[:, coef_i, eq_i])
                gt_mean = np.mean(gt_coef_array[eq_i, coef_i, :])
                gt_std = np.std(gt_coef_array[eq_i, coef_i, :])
                missed_coef_data.append([
                    f'{eqs[eq_i].split("=")[0]} : {coef_names[coef_i]}',
                    f'{gt_mean:.3f}', f'{est_mean:.3f}',
                    f'{gt_std:.3f}', f'{est_std:.3f}'
                    # f'{est_mean:.3f}', f'{est_std:.3f}',
                    # f'{gt_mean:.3f}', f'{gt_std:.3f}'
                ])

    ploted_eqs = []
    plot_counter = 0
    used_axes = []
    fig = plt.figure(layout="constrained",figsize=(figwidth,fighigth))
    # Loop to create and populate the subplots for the specified indices
    for index in to_plot:
        row_plot_idx = plot_counter // n_cols
        col = plot_counter % n_cols

        # Add a row title if it's a new row
        if index[0] not in ploted_eqs:
            # title_ax = fig.add_subplot(gs[1 * row_plot_idx, 0:n_cols])
            # title_ax.set_title(f'{eqs[index[0]]}', fontsize=12, fontweight='bold')
            # title_ax.axis('off')
            ploted_eqs.append(index[0])

        # Plot in the dedicated subplot row
        # axi = fig.add_subplot(gs[1 * row_plot_idx + 1, col])
        axi = plt.subplot2grid((n_rows, n_cols + 1),(row_plot_idx,col),colspan=1,fig=fig)
        # used_axes.append(axi)

        try:
            sns.kdeplot(est_coef_mean_arr[:, index[1], index[0]], ax=axi, fill=True, color=est_color, alpha=.4,
                        warn_singular=False, linewidth=3)
            if true_params:
                sns.kdeplot(gt_coef_array[index[0], index[1], :], ax=axi, fill=True, color=gt_color, alpha=.4,
                            warn_singular=False, linewidth=1)
        except:
            sns.kdeplot(est_coef_mean_arr[:, index[1], index[0]], ax=axi, fill=True, color=est_color, alpha=.4,
                        warn_singular=False, linewidth=3)
            if true_params:
                sns.kdeplot(gt_coef_array[index[0], index[1], :], ax=axi, fill=True, color=gt_color, alpha=.2,
                            warn_singular=False, linewidth=1)

        est_mean = np.mean(est_coef_mean_arr[:, index[1], index[0]])
        est_std = np.std(est_coef_mean_arr[:, index[1], index[0]])
        if true_params:
            gt_mean = np.mean(gt_coef_array[index[0], index[1], :])
            gt_std = np.std(gt_coef_array[index[0], index[1], :])

        axi.axvline(est_mean, color=est_color, linestyle='--', label=f'est_mean = {est_mean:.3f}')
        if true_params:
            axi.axvline(gt_mean, color=gt_color, linestyle='--', label=f'gt_mean = {gt_mean:.3f}')

        axi.set_xlabel(f"{eqs[eq_i]}\ncoef : {coef_names[index[1]]}", fontsize=xlabel_fontsize)
        axi.set_ylabel(" ", fontsize=xlabel_fontsize)
        axi.xaxis.set_major_locator(plt.MaxNLocator(3))
        axi.set_yticks(np.array([0]))
        axi.set_yticklabels([""], fontsize=12, rotation=90)
        axi.yaxis.set_tick_params(length=0)
        axi.spines['left'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.spines['right'].set_visible(False)
        if legend_True:
            axi.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=1, fancybox=True, shadow=True, fontsize=8)

        plot_counter += 1
        axi.set_xlim(0,2)

    # Hide unused subplots
    for i in range(plot_counter, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        print(row,col)
        axi = plt.subplot2grid((n_rows, n_cols + 1), (row, col), colspan=1,fig=fig)
        axi.axis('off')

    # Add the table to the last column, spanning all 2*n_rows
    ax_table = plt.subplot2grid((n_rows, n_cols + 1), (0, n_cols), colspan=1, rowspan=n_rows,fig=fig)
    ax_table.axis('off')

    if missed_coef_data:
        # Define table headers
        # headers = ['Eq. : Coef.', 'Est.\nMean', 'Est.\nStd', 'GT\nMean', 'GT\nStd']
        headers = ['Eq. : Coef.', 'GT\nMean', 'Est.\nMean', 'GT\nStd', 'Est.\nStd']
        # Create the table
        col_widths = [0.4, 0.15, 0.15, 0.15, 0.15]
        table = ax_table.table(cellText=missed_coef_data,
                               colLabels=headers,
                               loc='center',
                               cellLoc='center',
                               colWidths=col_widths)

        for i in range(len(missed_coef_data) + 1):
            for j in range(len(headers)):
                cell = table.get_celld()[(i, j)]
                if i == 0 or j == 0:
                    cell.set_facecolor('grey')
                else:
                    cell.set_facecolor('white')

        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1, 1.5)
        ax_table.set_title('Other Coefficients', fontsize=12)

    # Use tight_layout to handle spacing between GridSpec cells
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'row_kde_plot_with_table.jpg'))
    plt.show()
    print("plot run finished")