
################## row plot ######################

import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from scipy.stats import gaussian_kde



# inputs:
    # which indices(coef, eq),
    # how many row,col,
    # (gt, BH, Flat_B)
    # with or without table?

def Custom_plot(generate_pdf, ground_truth = True, HB_Est = True, FlatB_Est = True,pdf_state = True, TABLE = False, gt_utils=None, realparame2gtarray=None, save_dir_prefix=None,
               fighigth=3, figwidth=12, n_rows=None, n_cols=None,
               scaler=None, to_plot=None, plot_dict=dict()):
    if TABLE:
        all_cols = n_cols+2
    else:
        all_cols = n_cols

    root_path = os.getcwd()
    true_params_file_str = f"chk_GT_Data.pkl"
    save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith(save_dir_prefix)][0])

    legend_True = True if plot_dict is not None and plot_dict.get('legend') else None
    xlabel_fontsize = plot_dict.get('xlabel_fontsize', None)
    title_fontsize = plot_dict.get('title_fontsize', None)
    est_color = plot_dict.get('est_color', 'blue')
    gt_color = plot_dict.get('gt_color', 'pink')
    flat_color = plot_dict.get('flat_color', 'grey')
    pdf_color = plot_dict.get("pdf_color","red")
    pdf_fill = plot_dict.get("pdf_fill",False)
    pdf_line_width = plot_dict.get("pdf_line_width",2)
    max_y_limit = plot_dict.get("max_y_limit",0.98)
    plot_name = plot_dict.get("plot_name",'custom_plot.jpg')
    ############################# PDF #####################################
    if pdf_state:
        pdf_arr = generate_pdf(save_path)

    ############################## Load GT Data ###########################
    true_params_filename = os.path.join(save_path, true_params_file_str)
    with open(true_params_filename, 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)

    gt_coef_array = realparame2gtarray(true_params)
    gt_dict = gt_utils(true_params)
    eqs = gt_dict['eqs']
    coef_names = gt_dict['coef_names']
    N_Indv, N_Coef, N_Obs = X_data.shape
    N_Indv, N_Eqs, N_Obs = Y_data.shape
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

    print(f"est_coef_array.shape = {est_coef_array.shape}")
    est_coef_mean_arr = np.mean(est_coef_array, axis=0)
    ########################## Load Flat EST data #############################
    flat_save_path = os.path.join(root_path, [file for file in os.listdir(root_path) if file.startswith('Flat_'+save_dir_prefix)][0])
    if scaler is None:

        npz_sample_path = os.path.join(flat_save_path, "mcmc_samples.npz")
        loaded_samples = np.load(npz_sample_path, allow_pickle=True)
        flat_est_coef = loaded_samples['coef']
        flat_est_coef_array = np.array(flat_est_coef)
    else:
        with open(os.path.join(flat_save_path, f'revert_mcmc_samples.pkl'), 'rb') as f:
            flat_mcmc_coef_results = pickle.load(f)
            flat_est_coef_array = np.array(flat_mcmc_coef_results)

    print(f"est_coef_array.shape = {flat_est_coef_array.shape}")
    flat_est_coef_mean_arr = np.mean(flat_est_coef_array, axis=0)

    ################### Prepare plot data and table data ##################

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

                flatest_mean = np.mean(flat_est_coef_array[:, coef_i, eq_i])
                flatest_std = np.std(flat_est_coef_array[:, coef_i, eq_i])

                gt_mean = np.mean(gt_coef_array[eq_i, coef_i, :])
                gt_std = np.std(gt_coef_array[eq_i, coef_i, :])
                missed_coef_data.append([
                    f'{eqs[eq_i].split("=")[0]} : {coef_names[coef_i]}',
                    f'{gt_mean:.3f}', f'{est_mean:.3f}',f'{flatest_mean:.3f}',
                    f'{gt_std:.3f}', f'{est_std:.3f}', f'{flatest_std:.3f}'
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
        axi = plt.subplot2grid((n_rows, all_cols),(row_plot_idx,col),colspan=1,fig=fig)
        # used_axes.append(axi)
        y_lim = 0
        try:
            if HB_Est:
                print("HB add")
                sns.kdeplot(est_coef_mean_arr[:, index[1], index[0]], ax=axi, fill=False, color=est_color, alpha=.4,
                            warn_singular=False, linewidth=4,ls='--',label="Hierarchical Bayesian")
                data = est_coef_mean_arr[:, index[1], index[0]]
                kernel = gaussian_kde(data)
                x_values = np.linspace(min(data), max(data), 100)
                y_values = kernel(x_values)
                kernel = gaussian_kde(data)
                peak_y = np.max(y_values)
                y_lim = max(y_lim,peak_y)


            if ground_truth:
                print("gt add")
                if true_params:
                    sns.kdeplot(gt_coef_array[index[0], index[1], :], ax=axi, fill=False, color=gt_color, alpha=.8,
                                warn_singular=False, linewidth=1,label="Ground Truth")

                    data = gt_coef_array[index[0], index[1], :]
                    kernel = gaussian_kde(data)
                    x_values = np.linspace(min(data), max(data), 100)
                    y_values = kernel(x_values)
                    kernel = gaussian_kde(data)
                    peak_y = np.max(y_values)
                    y_lim = max(y_lim, peak_y)
            if pdf_state:
                print("pdf add")
                print(pdf_arr.shape)
                sns.kdeplot(pdf_arr[index[0], index[1],:], ax=axi, fill=pdf_fill, color=pdf_color, alpha=.6,
                            warn_singular=False, linewidth=4,label="Generator Distribution PDF")
                # data = pdf_arr[index[0], index[1],:]
                # kernel = gaussian_kde(data)
                # x_values = np.linspace(min(data), max(data), 100)
                # y_values = kernel(x_values)
                # kernel = gaussian_kde(data)
                # peak_y = np.max(y_values)
                # y_lim = max(y_lim, peak_y)

        except:
            if HB_Est:
                print("HB add")
                sns.kdeplot(est_coef_mean_arr[:, index[1], index[0]], ax=axi, fill=False, color=est_color, alpha=.4,
                            warn_singular=False, linewidth=4,ls='--',label="Hierarchical Bayesian")
                data = est_coef_mean_arr[:, index[1], index[0]]
                kernel = gaussian_kde(data)
                x_values = np.linspace(min(data), max(data), 100)
                y_values = kernel(x_values)
                kernel = gaussian_kde(data)
                peak_y = np.max(y_values)
                y_lim = max(y_lim,peak_y)

            if ground_truth:
                print("gt add")
                if true_params:
                    sns.kdeplot(gt_coef_array[index[0], index[1], :], ax=axi, fill=False, color=gt_color, alpha=.8,
                                warn_singular=False, linewidth=1,label="Ground Truth")
                    data = gt_coef_array[index[0], index[1], :]
                    kernel = gaussian_kde(data)
                    x_values = np.linspace(min(data), max(data), 100)
                    y_values = kernel(x_values)
                    kernel = gaussian_kde(data)
                    peak_y = np.max(y_values)
                    y_lim = max(y_lim, peak_y)

            if pdf_state:
                print("pdf add")
                sns.kdeplot(pdf_arr[index[0], index[1],:], ax=axi, fill=False, color=pdf_color, alpha=.4,
                            warn_singular=False, linewidth=4,label="Generator Distribution PDF")
                data = pdf_arr[index[0], index[1],:]
                kernel = gaussian_kde(data)
                x_values = np.linspace(min(data), max(data), 100)
                y_values = kernel(x_values)
                kernel = gaussian_kde(data)
                peak_y = np.max(y_values)
                y_lim = max(y_lim, peak_y)

        if FlatB_Est:
            print("flat add")
            print("flat shape",flat_est_coef_array.shape)
            sns.kdeplot(flat_est_coef_array[:, index[1],index[0]], ax=axi, fill=False, color=flat_color, alpha=.4,
                        warn_singular=False, linewidth=2,ls='-',label="Flat Bayesian")
            # max_y_limit = .98  # A good starting value, adjust as needed
        axi.set_ylim(0, y_lim * 1.2)

        # est_mean = np.mean(est_coef_mean_arr[:, index[1], index[0]])
        # est_std = np.std(est_coef_mean_arr[:, index[1], index[0]])
        # if true_params:
        #     gt_mean = np.mean(gt_coef_array[index[0], index[1], :])
        #     gt_std = np.std(gt_coef_array[index[0], index[1], :])
        #
        # axi.axvline(est_mean, color=est_color, linestyle='--', label=f'est_mean = {est_mean:.3f}')
        # if true_params:
        #     axi.axvline(gt_mean, color=gt_color, linestyle='--', label=f'gt_mean = {gt_mean:.3f}')

        axi.set_xlabel(f"{eqs[eq_i].split("=")[0]}\n{coef_names[index[1]]}", fontsize=xlabel_fontsize)
        axi.set_ylabel(" ", fontsize=xlabel_fontsize)
        axi.xaxis.set_major_locator(plt.MaxNLocator(3))
        axi.set_yticks(np.array([0]))
        axi.set_yticklabels([""], fontsize=12, rotation=90)
        axi.yaxis.set_tick_params(length=0)
        axi.spines['left'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.spines['right'].set_visible(False)
        if col==n_cols-1 and row_plot_idx==0:
            axi.legend(loc='center left', bbox_to_anchor=(.9, 1.03))
        if legend_True:
            axi.legend(loc='upper center', bbox_to_anchor=(0.9, 1.03), ncol=1, fancybox=True, shadow=True, fontsize=8)

        plot_counter += 1

    # Hide unused subplots
    for i in range(plot_counter, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        print(row,col)
        axi = plt.subplot2grid((n_rows, all_cols), (row, col), colspan=1,fig=fig)
        axi.axis('off')

    # Add the table to the last column, spanning all 2*n_rows
    if TABLE:
        ax_table = plt.subplot2grid((n_rows, all_cols), (0, n_cols), colspan=2, rowspan=n_rows,fig=fig)
        ax_table.axis('off')

        if missed_coef_data:
            # Define table headers
            # headers = ['Eq. : Coef.', 'Est.\nMean', 'Est.\nStd', 'GT\nMean', 'GT\nStd']
            headers = ['Eq. : Coef.', 'GT\nMean', 'HB.\nMean','Flat\nMean', 'GT\nStd', 'HB.\nStd','Flat\nStd']
            # Create the table
            col_widths = [0.4, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
            table = ax_table.table(cellText=missed_coef_data,
                                   colLabels=headers,
                                   loc='center',
                                   cellLoc='center',
                                   colWidths=col_widths)
            for j in range(len(headers)):
                cell = table.get_celld()[(0, j)]
                cell.set_height(0.1)

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
            ax_table.set_title('Other Coefficients',y=1.2, fontsize=12)

    # Use tight_layout to handle spacing between GridSpec cells
    # plt.tight_layout(w_pad=2, h_pad=2)
    plt.subplots_adjust(
        left=0.05,  # Left margin
        right=0.9,  # Right margin (pulls subplots left)
        bottom=0.2,  # Bottom margin (pushes subplots up)
        wspace=0.6  # Width space between subplots
    )
    plt.savefig(os.path.join(os.getcwd(), plot_name))
    plt.show()
    print("plot run finished")