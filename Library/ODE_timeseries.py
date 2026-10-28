import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# =====================================================================
# --- 1. NICHE DISPLACEMENT HELPER FUNCTIONS ---
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

# =====================================================================
# --- 2. ODE SYSTEM (Per-kg Scaling) ---
# =====================================================================

def NicheModel(t, y, p):
    y = np.maximum(y, 0.0)
    HSC, Neut, LSC, Blast = y
    
    total_stem = HSC + LSC
    if total_stem > p['niche_capacity']:
        scale = p['niche_capacity'] / total_stem
        HSC *= scale
        LSC *= scale
        total_stem = p['niche_capacity']
    
    s = 1.0 / (1.0 + p['k'] * Neut)
    
    empty_spaces = max(p['niche_capacity'] - total_stem, 0.0)
    p_empty = empty_spaces / p['niche_capacity']
    
    p_occ_H = HSC / p['niche_capacity']
    p_occ_L = LSC / p['niche_capacity']

    p_occ_L_total = p_empty + p_occ_H * p['p_HSCbyLSC']
    p_lodge_L = 1.0 - (1.0 - min(p_occ_L_total, 1.0))**p['tilde_n']

    p_occ_H_total = p_empty + p_occ_L * p['p_LSCbyHSC']
    p_lodge_H = 1.0 - (1.0 - min(p_occ_H_total, 1.0))**p['tilde_n']

    tmp1 = HSC_pushed_out(p_lodge_L, p['p1l'], LSC, p_occ_H, p['p_HSCbyLSC'], p_empty, p['tilde_n'])
    tmp2 = LSC_pushed_out(p_lodge_H, p['p1'] * s, HSC, p_occ_L, p['p_LSCbyHSC'], p_empty, p['tilde_n'])

    drug_on = p.get('drug_on', 0.0)
    kill_LSC = drug_on * p.get('ven_kill_rate_LSC', 0.0) * LSC
    kill_Blast = drug_on * p.get('ven_kill_rate_Blast', 0.0) * Blast
    kill_HSC = drug_on * p.get('ven_kill_rate_HSC', 0.0) * HSC
    kill_Neut = drug_on * p.get('ven_kill_rate_Neut', 0.0) * Neut

    dy0 = p_lodge_H * p['p1'] * s * HSC - tmp1 - p['d1'] * HSC - kill_HSC
    flux_h = (1.0 - p_lodge_H) * p['p1'] * s * HSC + tmp1
    dy1 = flux_h * (1.0 / (1.0 + p['gamma'] * Blast)) - p['d4'] * Neut - kill_Neut

    dy2 = p_lodge_L * p['p1l'] * LSC - tmp2 - kill_LSC
    flux_l = (1.0 - p_lodge_L) * p['p1l'] * LSC + tmp2 
    dy3 = flux_l - p['d4l'] * Blast - kill_Blast

    return [dy0, dy1, dy2, dy3]

# =====================================================================
# --- 3. DYNAMIC INITIAL DIAGNOSIS STATE CALCULATOR ---
# =====================================================================

def get_diagnosis_state(p_model, target_blast_frac, initial_neut_obs=None):
    p_untreated = p_model.copy()
    p_untreated['drug_on'] = 0.0

    hsc_init = p_untreated['niche_capacity'] - 1.0
    lsc_init = 1.0
    
    # Initialize pre-disease state with healthy steady-state neutrophils
    neut_init = (p_untreated['p1'] * hsc_init) / p_untreated['d4']
    y0_init = [hsc_init, neut_init, lsc_init, 0.0]

    def reach_diagnosis(t, y, p_arg):
        h, n, l, b = np.maximum(y, 0.0)
        blast_frac = (l + b) / (h + l + b + 1e-12)
        return blast_frac - target_blast_frac

    reach_diagnosis.terminal = True
    reach_diagnosis.direction = 1

    sol = solve_ivp(
        fun=NicheModel,
        t_span=(0, 10000),
        y0=y0_init,
        args=(p_untreated,),
        events=reach_diagnosis,
        method='LSODA',
        rtol=1e-3,
        atol=1e-6
    )

    if sol.success and len(sol.t_events[0]) > 0:
        return sol.y[:, -1]
    else:
        # Fallback to static distribution matching diagnosis measurements
        return [
            p_untreated['niche_capacity'] * (1.0 - target_blast_frac),
            initial_neut_obs if initial_neut_obs is not None else neut_init,
            p_untreated['niche_capacity'] * target_blast_frac,
            0.0
        ]

# =====================================================================
# --- 4. PHASED SIMULATION ENGINE ---
# =====================================================================

def run_phased_simulation(min_t, max_t, z0, p, schedule):
    boundaries = {min_t, max_t}
    for start, stop in schedule:
        if min_t <= start <= max_t: boundaries.add(start)
        if min_t <= stop <= max_t: boundaries.add(stop)
    boundaries = sorted(list(boundaries))
    
    t_all, y_all = [], []
    current_z0 = z0
    
    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i+1]
        if t0 == t1: continue
            
        mid_t = (t0 + t1) / 2.0
        is_drug_active = any(start <= mid_t <= stop for start, stop in schedule)
        
        p['drug_on'] = 1.0 if is_drug_active else 0.0
        
        sol = solve_ivp(
            fun=NicheModel, t_span=(t0, t1), y0=current_z0, args=(p,),
            method='LSODA', rtol=1e-3, atol=1e-6
        )
        
        if not sol.success: return None, None
            
        t_all.append(sol.t)
        y_all.append(sol.y)
        current_z0 = sol.y[:, -1]
        
    if not t_all: return None, None
    t_final = np.concatenate(t_all)
    y_final = np.concatenate(y_all, axis=1)
    
    t_unique, idx_unique = np.unique(t_final, return_index=True)
    return t_unique, y_final[:, idx_unique]

# =====================================================================
# --- 5. OBJECTIVE FUNCTION & OPTIMIZATION ENGINE ---
# =====================================================================

def objective_function(fitting_params, phsc_weighted, coef_weighted, base_params, clin_blood, clin_bm, schedule, range_neut, range_blast):
    drug_kill_LSC, drug_kill_HSC = fitting_params
    if drug_kill_LSC < 0 or drug_kill_HSC < 0: return 1e9
    
    try:
        p = base_params.copy()
        p['p1l'] = p['p1'] * coef_weighted
        p['p_LSCbyHSC'] = 1.0 - phsc_weighted
        p['p_HSCbyLSC'] = phsc_weighted
        
        p['ven_kill_rate_Blast'] = drug_kill_LSC
        p['ven_kill_rate_LSC']   = drug_kill_LSC
        p['ven_kill_rate_HSC']   = drug_kill_HSC
        p['ven_kill_rate_Neut']  = drug_kill_HSC
        
        initial_neut = clin_blood["b_neut"].iloc[0]
        initial_blast_perc = clin_bm["bm_blasts"].iloc[0] / 100.0

        z0 = get_diagnosis_state(p, initial_blast_perc, initial_neut)
        
        min_t = min(clin_blood['days_lab'].min(), clin_bm['days_from_bm'].min(), 0)
        max_t = max(clin_blood['days_lab'].max(), clin_bm['days_from_bm'].max()) + 10
        
        t_sim, y_sim = run_phased_simulation(min_t, max_t, z0, p, schedule)
        if t_sim is None: return 1e9
        
        sim_neut_interp = np.interp(clin_blood['days_lab'].values, t_sim, y_sim[1])
        scaled_diff_neut = (sim_neut_interp - clin_blood['b_neut'].values) / range_neut
        rmse_neut = np.sqrt(np.mean(scaled_diff_neut**2))
        
        sim_blast_perc = 100 * (np.interp(clin_bm['days_from_bm'].values, t_sim, y_sim[2]) + 
                               np.interp(clin_bm['days_from_bm'].values, t_sim, y_sim[3])) / \
                               (np.interp(clin_bm['days_from_bm'].values, t_sim, y_sim[0]) + 
                                np.interp(clin_bm['days_from_bm'].values, t_sim, y_sim[2]) + 
                                np.interp(clin_bm['days_from_bm'].values, t_sim, y_sim[3]) + 1e-12)
        scaled_diff_blast = (sim_blast_perc - clin_bm['bm_blasts'].values) / range_blast
        rmse_blast = np.sqrt(np.mean(scaled_diff_blast**2))
        
        return rmse_neut + rmse_blast
    except Exception:
        return 1e9

def optimize_patient_parameters(args):
    phsc_weighted, coef_weighted, base_params, clin_blood, clin_bm, schedule = args
    
    range_blast = 100.0
    range_neut = base_params.get('healthy_neut_per_kg', 4e8)
    
    try:
        initial_guess = [0.1, 0.01]
        res = minimize(
            objective_function,
            x0=initial_guess,
            method='Nelder-Mead',
            args=(phsc_weighted, coef_weighted, base_params, clin_blood, clin_bm, schedule, range_neut, range_blast),
            options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 1000, 'maxfev': 1500}
        )
        
        # Calculate cost at the final parameter state
        best_kill_LSC, best_kill_HSC = res.x
        total_error = objective_function(res.x, phsc_weighted, coef_weighted, base_params, clin_blood, clin_bm, schedule, range_neut, range_blast)
        
        # Accept if explicitly successful OR if valid finite parameters were found
        is_valid_fit = (res.success) or (total_error < 1e5 and best_kill_LSC > 0 and best_kill_HSC > 0)
        
        if not is_valid_fit:
            return None
        
        return {
            'Predefined_pHSC': phsc_weighted,
            'Predefined_Coef': coef_weighted,
            'Fitted_DrugKill_LSC': float(best_kill_LSC),
            'Fitted_DrugKill_HSC': float(best_kill_HSC),
            'RangeScaled_Combined_Error': float(total_error)
        }
    except Exception:
        return None

# =====================================================================
# --- 6. PLOT GENERATION FUNCTION ---
# =====================================================================

def generate_patient_fit_plots(atlas_df, neut_df, bm_df, cycles_df, base_params, output_dir="Plots"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for _, row in atlas_df.iterrows():
        patient_id = row['Patient_ID']

        try:
            blood_data = neut_df[neut_df['Pseudonym'] == patient_id].sort_values('days_lab').copy()
            if blood_data.empty: continue
            
            clin_neut_days = blood_data['days_lab'].values
            clin_neut_vals = blood_data['b_neut'].values

            bm_data = bm_df[bm_df['Pseudonym'] == patient_id].sort_values('days_from_bm').copy()
            if bm_data.empty: continue
            
            clin_blast_days = bm_data['days_from_bm'].values
            clin_blast_vals = bm_data['bm_blasts'].values

            drug_data = cycles_df[cycles_df['Pseudonym'] == patient_id].sort_values('venetoclax_start').copy()
            schedule = []
            for _, d_row in drug_data.iterrows():
                v_start = d_row.get('venetoclax_start')
                a_start = d_row.get('aza_start')
                v_stop = d_row.get('venetoclax_stop')
                a_stop = d_row.get('aza_stop')

                starts = [x for x in [v_start, a_start] if pd.notna(x)]
                stops = [x for x in [v_stop, a_stop] if pd.notna(x)]

                if starts and stops:
                    start = min(starts)
                    stop = max(stops)
                    if start <= stop:
                        schedule.append((start, stop))

            p = base_params.copy()
            phsc_weighted = row.get('pHSCbyLSC_weighted', row.get('Predefined_pHSC'))
            coef_weighted = row.get('coef_weighted', row.get('Predefined_Coef'))
            dk_LSC, dk_HSC = row['Fitted_DrugKill_LSC'], row['Fitted_DrugKill_HSC']

            p['p1l'] = p['p1'] * coef_weighted
            p['p_LSCbyHSC'] = 1.0 - phsc_weighted
            p['p_HSCbyLSC'] = phsc_weighted
            p['ven_kill_rate_Blast'] = dk_LSC
            p['ven_kill_rate_LSC']   = dk_LSC
            p['ven_kill_rate_HSC']   = dk_HSC
            p['ven_kill_rate_Neut']  = dk_HSC

            initial_neut = clin_neut_vals[0]
            initial_blast_perc = clin_blast_vals[0] / 100.0

            z0 = get_diagnosis_state(p, initial_blast_perc, initial_neut)
            
            min_t = min(clin_neut_days.min(), clin_blast_days.min(), 0)
            max_t = max(clin_neut_days.max(), clin_blast_days.max()) + 10

            t_sim, y_sim = run_phased_simulation(min_t, max_t, z0, p, schedule)
            if t_sim is None: continue

            sim_hsc, sim_neut, sim_lsc, sim_blast = y_sim
            sim_blast_perc = 100 * (sim_lsc + sim_blast) / (sim_hsc + sim_lsc + sim_blast + 1e-12)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, dpi=300)

            ax1.plot(t_sim, sim_neut, label="Simulated Neutrophils", color="blue")
            ax1.scatter(clin_neut_days, clin_neut_vals, color="black", s=45, label="Clinical Data")
            ax1.set_ylabel("Neutrophils (per kg)", fontsize=11)
            ax1.legend(loc="upper right")

            ax2.plot(t_sim, sim_blast_perc, label="Simulated Blasts %", color="red")
            ax2.scatter(clin_blast_days, clin_blast_vals, color="black", marker="s", s=40, label="Clinical Data")
            ax2.set_ylabel("Bone Marrow Blasts (%)", fontsize=11)
            ax2.set_xlabel("Time (Days)", fontsize=11)
            ax2.set_ylim(-5, 105)

            legend_shaded = False
            for start, stop in schedule:
                ax1.axvspan(start, stop, color="gray", alpha=0.25, zorder=1)
                ax2.axvspan(start, stop, color="gray", alpha=0.25, label="Drug Active" if not legend_shaded else "", zorder=1)
                legend_shaded = True

            ax2.legend(loc="upper right")
            fig.suptitle(f"Patient {patient_id} Multi-Target Fit\n(Fixed Priors | Fitted Kill LSC={dk_LSC:.4f}, Healthy={dk_HSC:.4f})", fontsize=12, y=0.96)
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            plt.savefig(out_path / f"Patient_{patient_id}_fit.png", bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"  -> Error rendering patient plot {patient_id}: {e}")
