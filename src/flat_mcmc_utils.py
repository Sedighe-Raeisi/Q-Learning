import numpy as np
import jax
from numpyro.infer import MCMC, NUTS
import time
import pickle
import os
from src.flat_svi_utils import svi_result_compare
from sklearn.preprocessing import StandardScaler
from src.model import pre_Flat
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
program_state = "start"


def run_mcmc(system_param_dict ,
             NUM_WARMUP, NUM_CHAINS, NUM_SAMPLES, NUM_BATCH_SAMPLES,
             model, mix_data , gt_utils,
             root_path = ".", save_dir_prefix = "chk_",
             program_state = "start",
             display_svi = True, scaler = None ):

    entries = os.listdir(root_path)
    BH_found_folders = []
    Flat_found_folders = []
    for entry in entries:
        if entry.startswith('Flat_'+save_dir_prefix):
            Flat_found_folders+=[entry]
        elif entry.startswith(save_dir_prefix):
            BH_found_folders+=[entry]
        # Check if it's a directory and starts with "chk_" (updated from "2025" for flexibility) #:
    if len(Flat_found_folders)!=0 and len(BH_found_folders)!=0:
        program_state = "continue_with_BH"
        FlatSavePath = os.path.join(root_path,Flat_found_folders[-1])
        BH_chk_path = os.path.join(root_path,BH_found_folders[-1])
    elif len(Flat_found_folders)!=0:
        program_state = "continue_without_BH"
        FlatSavePath = os.path.join(root_path,Flat_found_folders[-1])
        BH_chk_path = None
    elif len(BH_found_folders)!=0:  #:
        program_state = "start_with_BH"
        BH_chk_path = os.path.join(root_path,BH_found_folders[-1])
        FlatSavePath = os.path.join(root_path, 'Flat_'+BH_found_folders[-1])
        os.makedirs(FlatSavePath)
        print(f"----------- dir = {FlatSavePath} is created ----------")
    else:
        program_state = "start_without_BH"
        print("---------------------------- Make a directory for checkpoints and data -------------------------")
        time_stamp = time.strftime("%Y%m%d-%H%M")
        time_stamp = time_stamp.replace("-", "_")
        FlatSavePath = os.path.join(root_path, 'Flat_' + save_dir_prefix + time_stamp)
        os.makedirs(FlatSavePath)
        BH_chk_path = None
        print(f"----------- dir = {FlatSavePath} is created ----------")
    print(f"----------------------- program_state = (({program_state})) -----------------------")

    if program_state == "start_without_BH" or program_state == "start_with_BH":

        if program_state == "start_without_BH":
            true_params_filename = os.path.join(FlatSavePath, f"chk_GT_Data.pkl")
            X_data, Y_data, true_params = mix_data(system_param_dict)
        else:
            true_params_filename = os.path.join(BH_chk_path, f"chk_GT_Data.pkl")
            print(true_params_filename)
            with open(true_params_filename, 'rb') as f:
                X_data, Y_data, true_params = pickle.load(f)

        gt_dict = gt_utils(true_params)
        eq_list = gt_dict['eqs']
        coef_names = gt_dict['coef_names']
        gt_coef = gt_dict['gt_coef']
        if program_state == "start_without_BH":
            with open(os.path.join(FlatSavePath, "system_param_dict.pkl"), "wb") as f:
                pickle.dump(system_param_dict,f)
            with open(true_params_filename, 'wb') as f:
                pickle.dump((X_data, Y_data, true_params), f)
                print(f"--------------- GT Data saved as {true_params_filename} ------------")

        print("Shape of X_data:", X_data.shape)
        print("Shape of Y_data:", Y_data.shape)
        print("True parameters keys:", true_params.keys())
        ############################## if we need scaling #########################
        X_data, Y_data = pre_Flat(X_data, Y_data)
        if scaler is not None:
            scaler = BH_scaler(X_data)
            X_data = scaler.scale(X_data)


        if display_svi:

            ################## RUN SVI #####################################
            print("------------------ SVI Starts----------------------")

            svi_samples = svi_result_compare (X_data = X_data, Y_data = Y_data ,model=model,
                                      eqs = eq_list, coef_names = coef_names, gt_coef = gt_coef , scaler =scaler)

        print("------------------ MCMC Warmup Starts----------------------")

        master_rng_key = jax.random.PRNGKey(0)
        nuts_kernel = NUTS(model)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,  # Only run warmup, no sampling steps here
            num_chains=NUM_CHAINS,
            progress_bar=True,
        )

        run_rng_key, _ = jax.random.split(master_rng_key)
        mcmc.run(run_rng_key, X_data, Y_data)

        mcmc_samples = mcmc.get_samples()
        if scaler is not None:
            mcmc_coef_results = scaler.reverse_coef(mcmc_samples['coef'], method='mcmc')
            present_stationary_samples = mcmc.get_samples(group_by_chain=True)['coef']
            with open(os.path.join(FlatSavePath, f'revert_mcmc_samples.pkl'), 'wb') as f:
                pickle.dump(mcmc_coef_results, f)
        else:
            mcmc_coef_results = mcmc_samples['coef']
            present_stationary_samples = mcmc.get_samples(group_by_chain=True)['coef']
        print(f"the shape of mcmc ouput samples : {mcmc_coef_results.shape}")
        np.savez(os.path.join(FlatSavePath, f'mcmc_samples.npz'), **mcmc_samples)
        with open(os.path.join(FlatSavePath, f'mcmc_samples.pkl'), 'wb') as f:
            pickle.dump(mcmc_samples, f)
        print(f"----------------------- mcmc_samples.npz , mcmc_samples.pkl were created ----------------------")

        MCMC_PICKLE = "mcmc_module.pkl"
        with open(os.path.join(FlatSavePath, MCMC_PICKLE), "wb") as f:
            pickle.dump(mcmc, f)
        print(f"---------------- MCMC module saved to {MCMC_PICKLE} -----------------")
        print("----------------------- MCMC Warmup and first batch Finished -----------------------")
        mcmc_result(mcmc_coef_results, eq_list, coef_names, gt_coef)

        ############################ stability save ####################################
        file_list = os.listdir(FlatSavePath)
        if 'stationary_mcmc_samples.pkl' in file_list:
            with open(os.path.join(FlatSavePath, f'stationary_mcmc_samples.pkl'), 'rb') as f:
                stationary_mcmc_samples = pickle.load(f)
            stationary_mcmc_samples = np.concatenate((stationary_mcmc_samples,present_stationary_samples))
        else:
            print("------------------------- stationary sample file doesn't exist ------------------------")
            stationary_mcmc_samples = present_stationary_samples
        with open(os.path.join(FlatSavePath, f'stationary_mcmc_samples.pkl'), 'wb') as f:
            pickle.dump(stationary_mcmc_samples, f)
    else:
        if program_state == "continue_with_BH":
            true_params_filename = os.path.join(BH_chk_path, f"chk_GT_Data.pkl")
        elif program_state == "continue_without_BH":
            true_params_filename = os.path.join(FlatSavePath, f"chk_GT_Data.pkl")

        with open(true_params_filename, 'rb') as f:
            X_data, Y_data, true_params = pickle.load(f)
        gt_dict = gt_utils(true_params)
        eq_list = gt_dict['eqs']
        coef_names = gt_dict['coef_names']
        gt_coef = gt_dict['gt_coef']
        mcmc_pickle_path = os.path.join(FlatSavePath, "mcmc_module.pkl")
        with open(mcmc_pickle_path, "rb") as f:
            pickled_mcmc = pickle.load(f)

        pickled_mcmc.post_warmup_state = pickled_mcmc.last_state
        X_data, Y_data = pre_Flat(X_data, Y_data)
        if scaler is not None:
            scaler = BH_scaler(X_data)
            X_data = scaler.scale(X_data)

        pickled_mcmc.run(jax.random.PRNGKey(1), X_data, Y_data)

        mcmc_samples = pickled_mcmc.get_samples()

        if scaler is not None:
            mcmc_coef_results = scaler.reverse_coef(mcmc_samples['coef'], method='mcmc')
            present_stationary_samples = pickled_mcmc.get_samples(group_by_chain=True)['coef']
            with open(os.path.join(FlatSavePath, f'revert_mcmc_samples.pkl'), 'wb') as f:
                pickle.dump(mcmc_coef_results, f)
        else:
            mcmc_coef_results = mcmc_samples['coef']
            present_stationary_samples = pickled_mcmc.get_samples(group_by_chain=True)['coef']


        ############################ stability save ####################################
        file_list = os.listdir(FlatSavePath)
        if 'stationary_mcmc_samples.pkl' in file_list:
            with open(os.path.join(FlatSavePath, f'stationary_mcmc_samples.pkl'), 'rb') as f:
                stationary_mcmc_samples = pickle.load(f)
            stationary_mcmc_samples = np.concatenate((stationary_mcmc_samples,present_stationary_samples))
        else:
            print("------------------------- stationary sample file doesn't exist ------------------------")
            stationary_mcmc_samples = present_stationary_samples
        with open(os.path.join(FlatSavePath, f'stationary_mcmc_samples.pkl'), 'wb') as f:
            pickle.dump(stationary_mcmc_samples, f)

        #################################################################################
        np.savez(os.path.join(FlatSavePath, f'mcmc_samples.npz'), **mcmc_samples)
        with open(os.path.join(FlatSavePath, f'mcmc_samples.pkl'), 'wb') as f:
            pickle.dump(mcmc_samples, f)
        print(f"----------------------- mcmc_samples.npz , mcmc_samples.pkl were created ----------------------")

        MCMC_PICKLE = "mcmc_module.pkl"
        with open(os.path.join(FlatSavePath, MCMC_PICKLE), "wb") as f:
            pickle.dump(pickled_mcmc, f)
        print(f"---------------- MCMC module saved to {MCMC_PICKLE} -----------------")
        print("----------------------- MCMC continue batch Finished -----------------------")
        mcmc_result(mcmc_coef_results, eq_list, coef_names, gt_coef)
#######################################################################################################################################################
class BH_scaler:

    def __init__(self, X_data):
        self.X_data = X_data
        self.mean = None
        self.std = None

    def scale(self, X):
        """Scales the non-constant terms of the input data."""
        # X shape is (n_coef, n_obs)
        to_scale_X = X[1:, :]  # Exclude the constant term
        # Calculate mean and std across observations for each feature
        self.mean = np.mean(to_scale_X, axis=1, keepdims=True)
        self.std = np.std(to_scale_X, axis=1, keepdims=True)
        print("mean.shape",self.mean.shape)
        print("to_scale_X.shape",to_scale_X.shape)
        scaled_X = (to_scale_X - self.mean) / self.std
        print("scaled_X.shape",scaled_X.shape)
        # Reconstruct the scaled data with the constant term
        n_obs = X.shape[1]
        Theta_constant_term = np.ones((1, n_obs))
        print("Theta_constant_term.shape",Theta_constant_term.shape)
        scaled_X = np.concatenate((Theta_constant_term, scaled_X), axis=0)

        return scaled_X

    def reverse_coef(self, coef_arr, method='mcmc'):
        if method == 'mcmc':
            # mcmc output shape: (n_samples, n_coef, n_eq)
            const_coef = coef_arr[:, 0, :]
            nonConst_coef = coef_arr[:, 1:, :]

            # Reshape mean and std for broadcasting across samples and equations
            # The new mean and std have shape (1, n_coef - 1, 1)
            mean_reshaped = self.mean[np.newaxis, :, :]  # Shape becomes (1, n_coef - 1, 1)
            std_reshaped = self.std[np.newaxis, :, :]    # Shape becomes (1, n_coef - 1, 1)

            original_nonConst = nonConst_coef / std_reshaped
            original_const = const_coef - np.sum(original_nonConst * mean_reshaped, axis=1)

            original_coef = np.concatenate((np.expand_dims(original_const, axis=1), original_nonConst), axis=1)
            return original_coef

        elif method == 'svi':
            # svi output shape: (n_coef, n_eq) - assuming mean of posterior
            n_coef, n_eq = coef_arr.shape
            const_coef = coef_arr[0, :]
            nonConst_coef = coef_arr[1:, :]

            # Reshape mean and std for broadcasting across equations
            # The new mean and std have shape (n_coef - 1, 1)
            mean_reshaped = self.mean  # Shape is already (n_coef - 1, 1)
            std_reshaped = self.std    # Shape is already (n_coef - 1, 1)

            original_nonConst = nonConst_coef / std_reshaped
            original_const = const_coef - np.sum(original_nonConst * mean_reshaped, axis=0)

            original_coef = np.concatenate((np.expand_dims(original_const, axis=0), original_nonConst), axis=0)
            return original_coef

def mcmc_result(reverted_coef,eqs,coef_names,gt_coef):
    # Assuming 'reverted_coef' from the MCMC run is available
    N_samples, N_Coef_mcmc, N_Eqs_mcmc = reverted_coef.shape

    mean_est_list_mcmc = []
    std_est_list_mcmc = []

    for eq_j in range(N_Eqs_mcmc):
        mean_est_1eq = []
        std_est_1eq = []
        for coef_i in range(N_Coef_mcmc):
            # Calculate mean and std across samples and individuals for each coefficient
            est_mean = np.mean(reverted_coef[:, coef_i, eq_j])
            est_std = np.std(reverted_coef[:, coef_i, eq_j])
            mean_est_1eq.append(est_mean)
            std_est_1eq.append(est_std)
        mean_est_list_mcmc.append(mean_est_1eq)
        std_est_list_mcmc.append(std_est_1eq)

    mean_est_list_mcmc = np.array(mean_est_list_mcmc)
    std_est_list_mcmc = np.array(std_est_list_mcmc)

    print("--- MCMC Estimated Coefficients vs. Ground Truth ---")
    for eq_j in range(N_Eqs_mcmc):
        print('\n----------- Ground-truth Equation -----------\n')
        print(eqs[eq_j])
        print('\n------------- Recovered Equation (MCMC) -------------\n')
        for coef_i in range(N_Coef_mcmc):
            est_mean = mean_est_list_mcmc[eq_j, coef_i]
            est_std = std_est_list_mcmc[eq_j, coef_i]
            print(f"\n  For coef : (( {coef_names[coef_i]} )) we have")
            print(f"MCMC Est mean = {est_mean:.4f} , MCMC Est std = {est_std:.4f} ")
            print(
                f"GT mean = {gt_coef[eq_j][coef_names[coef_i]][0]:.4f} , GT std = {gt_coef[eq_j][coef_names[coef_i]][1]:.4f}")
