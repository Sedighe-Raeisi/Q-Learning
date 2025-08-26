import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# from model import MultiTargetMultiEquation_HSModel
import numpy as np
import jax
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer import  SVI, Trace_ELBO
from numpyro.optim import Adam
  # You can now use X_data and Y_data for Bayesian SINDy or other regression tasks
def run_svi(x,y,model, beta_names=None):
  guide = AutoNormal(model)
  svi = SVI(model, guide, Adam(0.01), loss=Trace_ELBO())
  PRNG_key = 1
  rng_key = jax.random.PRNGKey(PRNG_key)
  svi_result = svi.run(rng_key, 2000, xx=x, yy=y)
  vi_samples = svi_result.params
  return vi_samples

def svi_result_compare(X_data, Y_data, model, eqs, coef_names, gt_coef,scaler = None):
  svi_samples = run_svi(model=model, x=X_data, y=Y_data)
  # print("svi_samples.shape",svi_samples['coef_auto_loc'].shape)
  N_Eqs = svi_samples['coef_auto_loc'].shape[1]
  N_Coef = svi_samples['coef_auto_loc'].shape[0]
  if scaler is not None:
      svi_coef_result = scaler.reverse_coef(svi_samples['coef_auto_loc'],method='svi')
  else:
      svi_coef_result = svi_samples['coef_auto_loc']
  mean_est_list = []
  std_est_list = []

  for eq_j in range(N_Eqs):
      mean_est_1eq = []
      std_est_1eq = []
      for coef_i in range(N_Coef):
        est_mean = np.mean(svi_coef_result[ coef_i, eq_j])
        est_std = np.std(svi_coef_result[ coef_i, eq_j])
        mean_est_1eq += [est_mean]
        std_est_1eq += [est_std]
      mean_est_list += [mean_est_1eq]
      std_est_list += [std_est_1eq]
  mean_est_list = np.array(mean_est_list)
  std_est_list = np.array(std_est_list)

  # eqs = ["dx/dt = v", "dv/dt = -k/m * x - c/m * v + F0/m * cos(omega*t)"]
  # coef_names = ["constant", "x", "v", "cos(omega*t)", "x**2", "v**2", "x*v"]
  #
  # gt_coef = [{'constant':[0,0],'x':[0,0],'v':[1,0],'cos(omega*t)':[0,0],'x**2':[0,0],'v**2':[0,0],'x*v':[0,0]},
  #           {'constant':[0,0],'x':[true_params['-k/m_mean'],true_params['-k/m_std']],'v':[true_params['-c/m_mean'],true_params['-c/m_std']],'cos(omega*t)':[true_params['F0/m_mean'],true_params['F0/m_std']],'x**2':[0,0],'v**2':[0,0],'x*v':[0,0]}]
  print(f"the shape of svi ouput samples : {svi_samples['coef_auto_loc'].shape}")
  N_Eqs = svi_coef_result.shape[1]
  N_Coef = svi_coef_result.shape[0]
  print("------------------ SVI Results----------------------")

  for eq_j in range(N_Eqs):
      print('**************** Ground-truth Equation ****************')
      print(eqs[eq_j])
      print('**************** Recovered Equation ****************')
      for coef_i in range(N_Coef):
          est_mean = mean_est_list[eq_j,coef_i]
          est_std = std_est_list[eq_j,coef_i]
          print(f"====== For coef of (( {coef_names[coef_i]} )) we have ======")
          print(f"Est mean = {est_mean} , Est std = {est_std} ")
          print(f"GT mean = {gt_coef[eq_j][coef_names[coef_i]][0]} , GT std = {gt_coef[eq_j][coef_names[coef_i]][1]}")

  print("------------------ SVI Ends----------------------")
  return svi_coef_result