from Library import helper_functions
import numpy as np


def NicheModel(t, y, p):
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

    tmp1 = helper_functions.HSC_pushed_out(p_lodge_L, p['p1l'], LSC, p_occ_H, p['p_HSCbyLSC'], p_empty, p['tilde_n'])
    tmp2 = helper_functions.LSC_pushed_out(p_lodge_H, p['p1'] * s, HSC, p_occ_L, p['p_LSCbyHSC'], p_empty, p['tilde_n'])
    # ---------------------------------------------------------
    dy0 = p_lodge_H * p['p1'] * s * HSC - tmp1 - p['d1'] * HSC
    flux_h = (1.0 - p_lodge_H) * p['p1'] * s * HSC + tmp1
    
    dy1 = flux_h * (1.0 / (1.0 + p['gamma'] * Blast)) - p['d4'] * Neut

    dy2 = p_lodge_L * p['p1l'] * LSC - tmp2 
    flux_l = (1.0 - p_lodge_L) * p['p1l'] * LSC + tmp2 
    dy3 = flux_l - p['d4l'] * Blast

    return [dy0, dy1, dy2, dy3]
