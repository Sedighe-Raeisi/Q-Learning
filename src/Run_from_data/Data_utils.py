import numpy as np
import time
from typing import Callable
import pandas as pd
import math
import pickle
import os

# from src.Dynamical_systems_utils.Cognetive_RL.Daniel_Code.datasize_analysis import forget_rate
from src.utils import gen_param

############################### Mix data ######################################
def mix_data(system_param_dict):
    data_path = system_param_dict["data_path"]
    with open(data_path, 'rb') as f:
        XMixedData_np, YMixedData_np,  real_params= pickle.load(f)

    print("===============Preparing mix data is complete===============")
    # real_params = None



    return XMixedData_np, YMixedData_np, real_params


def gt_utils(real_params):
    eqs = ["QA_t = QA_[t-1]+Action*α*(R_QA[t-1])+(1-Action)*γ*(Q0-QA[t-1])"]
    coef_names = ['QA_1','ActionA*(R-QA_1)','(1-ActionA)*(QA0-QA_1)','QA_1**2','QB_1','QA_1*QB_1'] # Updated coef_names for RLC

    gt_coef = [{'QA_1': [1, 0],
                'ActionA*(R-QA_1)': [real_params['alpha_mean'], real_params['alpha_std']],
                '(1-ActionA)*(QA0-QA_1)': [real_params['forgetrate_mean'], real_params['forgetrate_std']],
                'QA_1**2': [0, 0],
                'QB_1': [0, 0],
                'QA_1*QB_1': [0, 0]}]
    return {"eqs":eqs, "coef_names":coef_names,"gt_coef":gt_coef}

def realparame2gtarray(real_params:dict):

    QA_arr = np.ones_like(real_params['alpha_array'])
    alpha_arr = real_params['alpha_array']
    fr_arr = real_params['forgetrate_array']
    QA2_arr = np.zeros_like(real_params['forgetrate_array'])
    QB_arr = np.zeros_like(real_params['forgetrate_array'])
    QAQB_arr = np.zeros_like(real_params['forgetrate_array'])



    eq1_coef = [QA_arr,alpha_arr,fr_arr,QA2_arr,QB_arr,QAQB_arr]


    eqs = np.array([eq1_coef])
    return eqs

def generate_pdf(save_path, pdf_smaple_N=10000, epsilon = 0.01 ):
    sys_param_path = os.path.join(save_path,"system_param_dict.pkl")
    with open(sys_param_path,"rb")as f:
        system_param_dict = pickle.load(f)

    Alpha_mean = system_param_dict['Alpha_info']["Alpha_mean"] if "Alpha_mean" in list(system_param_dict["Alpha_info"].keys()) else None
    Alpha_std = system_param_dict['Alpha_info']["Alpha_std"] if "Alpha_std" in list(system_param_dict["Alpha_info"].keys()) else None


    ForgetRate_mean = system_param_dict['ForgetRate_info']["ForgetRate_mean"] if "ForgetRate_mean" in list(system_param_dict["ForgetRate_info"].keys()) else None
    ForgetRate_std = system_param_dict['ForgetRate_info']["ForgetRate_std"] if "ForgetRate_std" in list(system_param_dict["ForgetRate_info"].keys()) else None
    ForgetRate_zeroP = system_param_dict['ForgetRate_info']["zero_peak_portion"] if "zero_peak_portion" in list(
        system_param_dict["ForgetRate_info"].keys()) else None

    data_i = 0
    alpha_list = []
    fr_list = []
    Alpha_gen = gen_param(N_param = pdf_smaple_N,a_mean = Alpha_mean, a_std = Alpha_std)
    if ForgetRate_zeroP:
        print("start pdf with zero portion")
        FR_gen = gen_param(N_param=pdf_smaple_N, a_mean=ForgetRate_mean, a_std=ForgetRate_std,
                           zero_peak_portion=ForgetRate_zeroP)
    else:
        FR_gen = gen_param(pdf_smaple_N, ForgetRate_mean, ForgetRate_std)

    for param_set in range(pdf_smaple_N):
        alpha = math.fabs(Alpha_gen.gen())
        forget_rate = math.fabs(FR_gen.gen())
        alpha_list.append(alpha)
        fr_list.append(forget_rate)
        data_i += 1

    coef_list = [[np.abs(np.random.normal(1, epsilon, pdf_smaple_N)),  # const
                   np.array(alpha_list),  #
                   np.array(fr_list),  #
                   np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),  #
                   np.abs(np.random.normal(0, epsilon, pdf_smaple_N)),  #
                   np.abs(np.random.normal(0, epsilon, pdf_smaple_N))]]
    return np.array(coef_list)
