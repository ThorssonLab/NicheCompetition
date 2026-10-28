import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import multiprocessing
import pickle
from Library import ODE_diagnosis
from Library import helper_functions

def run_trajectory_grid(phsc_range, coef_range, base_params, gamma_val, y0_healthy, min_time=30):
    """
    Generates the internal simulator trajectories for a fixed gamma configuration.
    Unified Cohort Version: min_time is universally set to 30 days to capture 
    aggressive progression kinetics for both de novo and sAML/RR patients.
    """
    traj_list = []
    
    for phsc in phsc_range:
        for coef in coef_range:
            p = base_params.copy()
            p['gamma'] = gamma_val
            p['p1l'] = p['p1'] * coef
            p['p_LSCbyHSC'] = 1.0 - phsc
            p['p_HSCbyLSC'] = phsc
            
            z0 = [min(y0_healthy[0], p['niche_capacity'] - 1), y0_healthy[1], 1.0, 0.0]
            sol = solve_ivp(
                fun=ODE_diagnosis.NicheModel, 
                t_span=(0, 25000), 
                y0=z0, 
                args=(p,),
                method='LSODA', 
                events=helper_functions.hit_max_blasts, 
                rtol=1e-3, 
                atol=1e-6 )
            
            if sol.success or sol.status == 1:
                Y_out = sol.y.T
                BP = (Y_out[:, 2] + Y_out[:, 3]) / (Y_out[:, 0] + Y_out[:, 2] + Y_out[:, 3] + 1e-12)
                
                out_df = pd.DataFrame({
                    'pHSC': phsc, 'Coef': coef, 'Time': sol.t,
                    'Neut': Y_out[:, 1], 'BlastPerc': BP
                })
                # Apply Dynamic Biological Plausibility Filters
                reached_20 = out_df[out_df['BlastPerc'] >= 0.20]
                if not reached_20.empty:
                    diag_row = reached_20.iloc[0]
                    # Unified boundary: All patients (De novo and Relapse) share the 30-day minimum
                    if min_time <= diag_row['Time'] <= 5500 and diag_row['Neut'] >= 0:
                        traj_list.append(out_df)
    return traj_list
