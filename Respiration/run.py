import data, utils, processing, metric
import numpy as np

root = 'E:/Deep_Learning/Respiration/'
data_root = f"{root}DATA/"
result_root = f"{root}RESULTS/"

num_attempts = 2 # MTG_{number}

while True:
    proceed = int(input("Please enter '1' if you want to proceed, or enter '0': "))
    print()
    if proceed == 0: break
    treatment = input("Desired Treatment Method [Type 'STATIC' or 'ARC']: ")
    print()
    breath = input("Desired Breath Type: [Type 'Breathhold' or 'FULL']: ")
    print()

    patient_path, num_fx = data.patient_path(data_root, treatment, breath)
    patient_ID = patient_path.split("/")[5].split("_")[0]
    result_folder = f"{result_root}MTG_{num_attempts}/{patient_ID}/"
    utils.createFolder(result_folder)
    with open(f"{result_folder}metric.txt", "w") as f:
        patient_path, num_fx = data.patient_path(data_root, treatment, breath)
        patient_ID = patient_path.split("/")[5].split("_")[0]
        RPD_per_fld, STB_per_fld = [], [] # Metric over entire session (each metric is calculated for each field)
        RPD_per_fx, STB_per_fx = [], [] # Metric over entire session (each metric is calculated for each fraction)
        for fx in range(1, num_fx+1):
            f.write(f"\t\t\t\t=====Fraction{fx}=====\n\n")
            fraction_path, num_fld = data.fraction_path(patient_path, fx)
            rpds_per_fld, stbs_per_fld = [], [] # Reproducibility & Stability over each field -> [fld_rpd1, fld_rpd2, ...]
            fx_lvls, fx_errors = [], [] # Avg_lvl & Vert_error for each fraction (over ALL fields) -> [avg_lvl1, avg_lvl2, ...]
            for field in range(1, num_fld+1):
                f.write(f"\t\t\t\t =====Field{field}=====\n")
                (data_Times, data_Amps), (beam_Times, beam_Amps) = data.read_field_data(fraction_path, field)
                dilated_beams = processing.dilate_beams(data_Times, beam_Times, beam_Amps)
                cutted_amps = np.array(data_Amps) * np.array(dilated_beams)
                enabled_intervals, num_intvs = processing.beam_enabling_intervals(data_Times, cutted_amps)
                field_lvls, field_errors = [], []
                for intv in range(num_intvs):
                    avg_lvl = metric.avg_lvl_per_interval(enabled_intervals[intv]) * 10 # cm -> mm
                    error = metric.error_per_interval(enabled_intervals[intv]) * 10     # cm -> mm
                    field_lvls.append(avg_lvl)
                    field_errors.append(error)
                    fx_lvls.append(avg_lvl)
                    fx_errors.append(error)
                    f.write(f"[Interval{intv}] Average Level (mm): {avg_lvl:.4f}\tVertical Error (mm): {error:.4f}\n")
                fld_reprod = metric.reprod_per_field(field_lvls)
                fld_stab = metric.stab_per_field(field_errors)
                rpds_per_fld.append(fld_reprod)
                stbs_per_fld.append(fld_stab)
                f.write(f"Reproducibility (mm): {fld_reprod:.4f}\tStability (mm): {fld_stab:.4f}\n\n")
            fx_mean_reprod = metric.mean_reprod_per_fraction(rpds_per_fld)
            fx_mean_stab = metric.mean_stab_per_fraction(stbs_per_fld)
            fx_rpd = metric.reprod_per_field(fx_lvls)
            fx_stb = metric.stab_per_field(fx_errors)
            RPD_per_fld.append(fx_mean_reprod)
            STB_per_fld.append(fx_mean_stab)
            RPD_per_fx.append(fx_rpd)
            STB_per_fx.append(fx_stb)
            f.write(f"Mean Reproducibility (mm): {fx_mean_reprod:.4f}\tMean Stability (mm): {fx_mean_stab:.4f}\n")
            f.write(f"Fraction Reproducibility (mm): {fx_rpd:.4f}\tFraction Stability (mm): {fx_stb:.4f}\n\n\n")
        R2_RPD, R2_STB = metric.R_squared(RPD_per_fx), metric.R_squared(STB_per_fx)
        CV_RPD, CV_STB = metric.coeff_var(RPD_per_fx), metric.coeff_var(STB_per_fx)
        f.write("\t\t\t\t=====R-squared=====\n")
        f.write(f"Reproducibility: {R2_RPD:.4f}\tStability: {R2_STB:.4f}\n\n")
        f.write("\t\t\t\t=====CV=====\n")
        f.write(f"Reproducibility: {CV_RPD:.4f}\tStability: {CV_STB:.4f}\n\n")
    
    print(f"Analysis on [{patient_ID}] is complete.")
    print()

print("Program terminated.")