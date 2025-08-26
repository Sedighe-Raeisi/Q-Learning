import os
import numpy as np
import arviz as az
import pandas as pd
from src.model import MultiTargetMultiEquation_HSModel, Flat_HSModel, pre_Flat
import pickle
from src.mcmc_utils import BH_scaler

def compare_model(hb_save_dir_prefix,scaler = None):
    """
    Loads MCMC results from two models, converts them to InferenceData,
    and compares them using az.compare_results, az.loo, and az.waic.
    """
    models = {}
    root_path = os.getcwd()
    flat_save_dir_prefix = "Flat_"+hb_save_dir_prefix
    # --- Process the Hierarchical Bayesian (HB) Model ---
    folder_path_hb = os.path.join(
        root_path,
        [file for file in os.listdir(root_path) if file.startswith(hb_save_dir_prefix)][0]
    )
    with open(os.path.join(folder_path_hb, "chk_GT_Data.pkl"), 'rb') as f:
        X_data, Y_data, true_params = pickle.load(f)
        if scaler is not None:
            scaler = BH_scaler(X_data)
            X_data = scaler.scale(X_data)
    with open(os.path.join(folder_path_hb, "mcmc_module.pkl"), "rb") as f:
        hb_loaded_mcmc = pickle.load(f)
    dims_hb = {"obs": ["id", "target", "time"]}
    idata_kwargs_hb = {"dims": dims_hb, "constant_data": {"xx": X_data}}
    idata_hb = az.from_numpyro(hb_loaded_mcmc, **idata_kwargs_hb)
    models['Hierarchical_Model'] = idata_hb

    # --- Process the Flat Bayesian (FlatB) Model ---
    folder_path_flat = os.path.join(
        root_path,
        [file for file in os.listdir(root_path) if file.startswith(flat_save_dir_prefix)][0]
    )
    with open(os.path.join(folder_path_flat, "mcmc_module.pkl"), "rb") as f:
        flat_loaded_mcmc = pickle.load(f)
    X_data_flat, Y_data_flat = pre_Flat(X_data, Y_data)
    dims_flat = {"obs": ["target", "time"]}
    idata_kwargs_flat = {"dims": dims_flat, "constant_data": {"xx": X_data_flat}}
    idata_flat = az.from_numpyro(flat_loaded_mcmc, **idata_kwargs_flat)
    models['Flat_Model'] = idata_flat

    # --- Compute and Compare Metrics ---
    print("\nComputing model comparison metrics...")

    # Use az.compare_results which already includes LOO and WAIC
    comparison_results = az.compare(models, ic='loo', method='stacking')

    # Computing other useful metrics
    hierarchical_loo = az.loo(idata_hb)
    flat_loo = az.loo(idata_flat)

    hierarchical_waic = az.waic(idata_hb)
    flat_waic = az.waic(idata_flat)

    # Combine results into a single DataFrame
    results_dict = {
        'Hierarchical_Model': {
            'loo': hierarchical_loo.elpd_loo,
            'loo_se': hierarchical_loo.se,
            'p_loo': hierarchical_loo.p_loo,
            'waic': hierarchical_waic.elpd_waic,
            'waic_se': hierarchical_waic.se,
            'p_waic': hierarchical_waic.p_waic
        },
        'Flat_Model': {
            'loo': flat_loo.elpd_loo,
            'loo_se': flat_loo.se,
            'p_loo': flat_loo.p_loo,
            'waic': flat_waic.elpd_waic,
            'waic_se': flat_waic.se,
            'p_waic': flat_waic.p_waic
        }
    }

    additional_metrics_df = pd.DataFrame(results_dict).T

    # Merge the comparison_results and the new metrics
    final_df = comparison_results.merge(additional_metrics_df, left_index=True, right_index=True)

    # Print and Save the Results
    print("\nComparison Results:")
    print(final_df)

    file_name = [file for file in os.listdir(root_path) if file.startswith(hb_save_dir_prefix)][0]

    # Save the combined DataFrame to a CSV file
    csv_file_path = os.path.join(root_path, "compare_"+file_name + ".csv")
    final_df.to_csv(csv_file_path)
    print(f"\nModel comparison results saved to {csv_file_path}")

    # Save the DataFrame to a pickle file as before
    pkl_file_path = os.path.join(root_path, "compare_"+file_name + ".pkl")
    with open(pkl_file_path, "wb") as f:
        pickle.dump(final_df, f)
    print(f"Model comparison results saved to {pkl_file_path}")