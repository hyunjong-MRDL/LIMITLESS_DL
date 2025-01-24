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

    with open(f"{result_folder}stats.txt", "w") as f:
        patient_path, num_fx = data.patient_path(data_root, treatment, breath)
        patient_ID = patient_path.split("/")[5].split("_")[0]
        total_lvls, total_errors = [], []

        for fx in range(1, num_fx+1):
            f.write(f"\t\t=====Fraction{fx}=====\n\n")
            fraction_path, num_fld = data.fraction_path(patient_path, fx)
            fraction_lvls, fraction_errors = [], []
            fx_lvl_CV, fx_err_CV = [], []

            for field in range(1, num_fld+1):
                f.write(f"\t\t=====Field{field}=====\n")
                f.write(f"\t  Average Levels\tVertical Errors\n")
                (data_Times, data_Amps), (beam_Times, beam_Amps) = data.read_field_data(fraction_path, field)
                cutted_Amps = processing.cut_by_beams(data_Times, data_Amps, beam_Times)
                enabled_intervals, num_intvs = processing.beam_enabling_intervals(data_Times, data_Amps, beam_Times)
                field_lvls, field_lines, field_errors = [], [], []

                for intv in range(num_intvs):
                    avg_lvl = metric.avg_lvl_per_interval(enabled_intervals[intv])
                    fitted_line = processing.regression_line(enabled_intervals[intv])
                    error = metric.error_per_interval(enabled_intervals[intv])
                    field_lvls.append(avg_lvl)
                    field_lines.append(fitted_line)
                    field_errors.append(error)

                fld_mean_lvls = np.mean(field_lvls)
                fld_std_lvls = np.std(field_lvls)
                fld_mean_errors = np.mean(field_errors)
                fld_std_errors = np.std(field_errors)
                fld_lvl_CV = fld_std_lvls/fld_mean_lvls
                fld_err_CV = fld_std_errors/fld_mean_errors
                f.write(f"Mean [mm]:\t{10*fld_mean_lvls:.4f}\t{10*fld_mean_errors:.4f}\n")  # cm -> mm
                f.write(f"STD [mm]:\t{10*fld_std_lvls:.4f}\t{10*fld_std_errors:.4f}\n")  # cm -> mm
                f.write(f"CV: \t{100*fld_lvl_CV:.4f}%\t{100*fld_err_CV:.4f}%\n")  # Show as percentage
                fraction_lvls.append(fld_mean_lvls)
                fraction_errors.append(fld_mean_errors)
                fx_lvl_CV.append(fld_lvl_CV)
                fx_err_CV.append(fld_err_CV)

            total_lvls.append(np.mean(fraction_lvls))
            total_errors.append(np.mean(fraction_errors))
            f.write(f"\t  Total metrics for Fraction{fx}\n")
            f.write(f"CV: \t{100*metric.coeff_var(fraction_lvls):.4f}%\t{100*metric.coeff_var(fraction_errors):.4f}%\n")
            f.write(f"Mean_CV:\t{100*np.mean(fx_lvl_CV):.4f}\t{100*np.mean(fx_err_CV):.4f}\n")  # cm -> mm
    
    print("\t  Average Levels\tVertical Errors\n")
    print(f"Mean throughout 30 fractions: {10*np.mean(total_lvls):.4f}\t{10*np.mean(total_errors):.4f}")
    print(f"CV: {100*metric.coeff_var(total_lvls):.4f}\t{100*metric.coeff_var(total_errors):.4f}")
    print(f"Analysis on [{patient_ID}] is complete.")
    print()

print("Program terminated.")