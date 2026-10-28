import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import multiprocessing
import pickle

def HSC_pushed_out(p_niche_found_LSC, p1l, LSC, p_occupied_by_HSC, p_HSC_by_LSC, p_empty, tilde_n):
    if p_empty == 0:
        return (1.0 - (1.0 - p_occupied_by_HSC * p_HSC_by_LSC)**tilde_n) * p1l * LSC
    elif p_HSC_by_LSC == 0:
        return 0.0
    else:
        return p_niche_found_LSC * p_occupied_by_HSC * p_HSC_by_LSC / (p_empty + p_occupied_by_HSC * p_HSC_by_LSC) * p1l * LSC

def LSC_pushed_out(p_niche_found_HSC, p1, HSC, p_occupied_by_LSC, p_LSC_by_HSC, p_empty, tilde_n):
    if p_empty == 0:
        return (1.0 - (1.0 - p_occupied_by_LSC * p_LSC_by_HSC)**tilde_n) * p1 * HSC
    elif p_LSC_by_HSC == 0:
        return 0.0
    else:
        return p_niche_found_HSC * p_occupied_by_LSC * p_LSC_by_HSC / (p_empty + p_occupied_by_LSC * p_LSC_by_HSC) * p1 * HSC

def hit_max_blasts(t, y, p):
    """Event triggers and stops solver when Blast Fraction hits 99.9%"""
    HSC, Neut, LSC, Blast = y
    sum_h = HSC
    sum_l = LSC + Blast
    BP = sum_l / (sum_h + sum_l + 1e-12)
    return BP - 0.999 

hit_max_blasts.terminal = True 
hit_max_blasts.direction = 1 

def NicheModel(t, y, p):
    # Guard against negative cell counts from numerical solver overshoot
    y = np.maximum(y, 0.0) 
    HSC, Neut, LSC, Blast = y
    
    total_stem = HSC + LSC
    if total_stem > p['niche_capacity']:
        scale = p['niche_capacity'] / total_stem
        HSC *= scale
        LSC *= scale
        total_stem = p['niche_capacity']
    
    # Proliferation feedback strictly depends on healthy mature cells
    s = 1.0 / (1.0 + p['k'] * Neut)
    
    empty_spaces = max(p['niche_capacity'] - total_stem, 0.0)
    p_empty = empty_spaces / p['niche_capacity']
    
    p_occ_H = HSC / p['niche_capacity']
    p_occ_L = LSC / p['niche_capacity']

    # Niche lodging probabilities 
    p_occ_L_total = p_empty + p_occ_H * p['p_HSCbyLSC']
    p_lodge_L = 1.0 - (1.0 - min(p_occ_L_total, 1.0))**p['tilde_n']

    p_occ_H_total = p_empty + p_occ_L * p['p_LSCbyHSC']
    p_lodge_H = 1.0 - (1.0 - min(p_occ_H_total, 1.0))**p['tilde_n']

    # External hooks for pushed out cells
    tmp1 = HSC_pushed_out(p_lodge_L, p['p1l'], LSC, p_occ_H, p['p_HSCbyLSC'], p_empty, p['tilde_n'])
    tmp2 = LSC_pushed_out(p_lodge_H, p['p1'] * s, HSC, p_occ_L, p['p_LSCbyHSC'], p_empty, p['tilde_n'])

    # ---------------------------------------------------------
    # System ODEs (diff_HSC and diff_LSC completely removed)
    # ---------------------------------------------------------
    dy0 = p_lodge_H * p['p1'] * s * HSC - tmp1 - p['d1'] * HSC
    flux_h = (1.0 - p_lodge_H) * p['p1'] * s * HSC + tmp1
    
    # Gamma (g) scales blast suppression effect on differentiation
    dy1 = flux_h * (1.0 / (1.0 + p['gamma'] * Blast)) - p['d4'] * Neut

    dy2 = p_lodge_L * p['p1l'] * LSC - tmp2 
    flux_l = (1.0 - p_lodge_L) * p['p1l'] * LSC + tmp2 
    dy3 = flux_l - p['d4l'] * Blast

    return [dy0, dy1, dy2, dy3]
