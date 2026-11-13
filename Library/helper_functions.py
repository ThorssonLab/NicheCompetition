def hit_max_blasts(t, y, p):
    """Event triggers and stops solver when Blast Fraction hits 99.9%"""
    HSC, Neut, LSC, Blast = y
    sum_h = HSC
    sum_l = LSC + Blast
    BP = sum_l / (sum_h + sum_l + 1e-12)
    return BP - 0.999 

hit_max_blasts.terminal = True 
hit_max_blasts.direction = 1 


# =====================================================================
# --- 1.ODE HELPER FUNCTIONS ---
# =====================================================================
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
