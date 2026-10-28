

def hit_max_blasts(t, y, p):
    """Event triggers and stops solver when Blast Fraction hits 99.9%"""
    HSC, Neut, LSC, Blast = y
    sum_h = HSC
    sum_l = LSC + Blast
    BP = sum_l / (sum_h + sum_l + 1e-12)
    return BP - 0.999 


hit_max_blasts.terminal = True 
hit_max_blasts.direction = 1 