import data, utils, processing, metric
import numpy as np
import pandas as pd
import pingouin as pg

root = 'E:/LIMITLESS_DL/Respiration/'
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

    #### 1. Beam analysis (now)
    with open(f"{result_folder}beam.txt", "w") as f:
        patient_path, num_fx = data.patient_path(data_root, treatment, breath)
        patient_ID = patient_path.split("/")[5].split("_")[0]
        total_RPDs, total_STBs = [], []
        total_lvl_CV, total_dist_CV = [], []
        total_lvl_ICC, total_dist_ICC = [], []

        for fx in range(1, num_fx+1):
            f.write(f"\t\t=====Fraction{fx}=====\n\n")
            fraction_path, num_fld = data.fraction_path(patient_path, fx)
            fx_levels, fx_errors = [], []

            for field in range(1, num_fld+1):
                f.write(f"\t\t=====Field{field}=====\n")
                (data_Times, data_Amps), (beam_Times, beam_Amps) = data.read_field_data(fraction_path, field)
                cutted_Amps = processing.cut_by_beams(data_Times, data_Amps, beam_Times)
                enabled_intervals, num_intvs = processing.beam_enabling_intervals(data_Times, data_Amps, beam_Times)
                beam_levels, beam_errors = [], []

                for intv in range(num_intvs):
                    intv_level = metric.avg_lvl_per_interval(enabled_intervals[intv])
                    intv_error = metric.error_per_interval(enabled_intervals[intv])
                    beam_levels.append(intv_level)
                    beam_errors.append(intv_error)
                
                # level_data = {
                #     "Subject": list(range(1, len(beam_levels)+1)),
                #     "Rater": ["beam"] * len(beam_levels),
                #     "Score": beam_levels
                # }
                # level_df = pd.DataFrame(level_data)
                # level_icc = pg.intraclass_corr(level_df, "Subject", "Rater", "Score")

                # dist_data = {
                #     "Subject": list(range(1, len(beam_errors)+1)),
                #     "Rater": ["beam"] * len(beam_errors),
                #     "Score": beam_errors
                # }
                # dist_df = pd.DataFrame(dist_data)
                # dist_icc = pg.intraclass_corr(dist_df, "Subject", "Rater", "Score")

                total_RPDs.append(metric.reprod_per_field(beam_levels))
                total_STBs.append(metric.stab_per_field(beam_errors))
                total_lvl_CV.append(metric.coeff_var(beam_levels))
                total_dist_CV.append(metric.coeff_var(beam_errors))
                # total_lvl_ICC.append(level_icc)
                # total_dist_ICC.append(dist_icc)

                f.write(f"\t  Average Levels\tVertical Errors\n")
                f.write(f"Mean [mm]:\t{10*np.mean(beam_levels):.5f}\t{10*np.mean(beam_errors):.5f}\n")  # cm -> mm
                f.write(f"STD [mm]:\t{10*np.std(beam_levels):.5f}\t{10*np.std(beam_errors):.5f}\n")  # cm -> mm
                f.write(f"CV: \t{100*metric.coeff_var(beam_levels):.5f}%\t{100*metric.coeff_var(beam_errors):.5f}%\n")  # Show as percentage
                # f.write(f"ICC: \t{level_icc:.5f}\t{dist_icc:.5f}\n")

                f.write(f"Reproducibility [mm]:\t{10*metric.reprod_per_field(beam_levels):.5f}\n")
                f.write(f"Stability [mm]:\t{10*metric.stab_per_field(beam_errors):.5f}\n\n")
    
    print(f"Reproducibility: {10*np.mean(total_RPDs):.5f}\n")
    print(f"Stability: {10*np.mean(total_STBs):.5f}\n\n")
    print(f"\t  Average Levels\tVertical Errors\n")
    print(f"CV: \t{100*np.mean(total_lvl_CV):.5f}\t{100*np.mean(total_dist_CV):.5f}\n")
    # print(f"ICC: \t{np.mean(total_lvl_ICC):.5f}\t{np.mean(total_dist_ICC):.5f}\n")
    print(f"Analysis on [{patient_ID}] is complete.\n\n\n\n")
    print()

    #### 2. Field analysis
    with open(f"{result_folder}field.txt", "w") as f:
        patient_path, num_fx = data.patient_path(data_root, treatment, breath)
        patient_ID = patient_path.split("/")[5].split("_")[0]
        total_RPDs, total_STBs = [], []
        total_lvl_CV, total_dist_CV = [], []
        total_lvl_ICC, total_dist_ICC = [], []

        for fx in range(1, num_fx+1):
            f.write(f"\t\t=====Fraction{fx}=====\n\n")
            fraction_path, num_fld = data.fraction_path(patient_path, fx)
            fld_levels, fld_errors = [], []

            for field in range(1, num_fld+1):
                (data_Times, data_Amps), (beam_Times, beam_Amps) = data.read_field_data(fraction_path, field)
                cutted_Amps = processing.cut_by_beams(data_Times, data_Amps, beam_Times)
                enabled_intervals, num_intvs = processing.beam_enabling_intervals(data_Times, data_Amps, beam_Times)

                for intv in range(num_intvs):
                    intv_level = metric.avg_lvl_per_interval(enabled_intervals[intv])
                    intv_error = metric.error_per_interval(enabled_intervals[intv])
                    fld_levels.append(np.mean(intv_level))
                    fld_errors.append(np.mean(intv_error))

            # level_data = {
            #         "Subject": list(range(1, len(fld_levels)+1)),
            #         "Rater": ["beam"] * len(fld_levels),
            #         "Score": fld_levels
            #     }
            # level_df = pd.DataFrame(level_data)
            # level_icc = pg.intraclass_corr(level_df, "Subject", "Rater", "Score")

            # dist_data = {
            #     "Subject": list(range(1, len(fld_errors)+1)),
            #     "Rater": ["beam"] * len(fld_errors),
            #     "Score": fld_errors
            # }
            # dist_df = pd.DataFrame(dist_data)
            # dist_icc = pg.intraclass_corr(dist_df, "Subject", "Rater", "Score")
            
            total_RPDs.append(metric.reprod_per_field(fld_levels))
            total_STBs.append(metric.stab_per_field(fld_errors))
            total_lvl_CV.append(metric.coeff_var(fld_levels))
            total_dist_CV.append(metric.coeff_var(fld_errors))
            # total_lvl_ICC.append(level_icc)
            # total_dist_ICC.append(dist_icc)

            f.write(f"\t  Average Levels\tVertical Errors\n")
            f.write(f"Mean [mm]:\t{10*np.mean(fld_levels):.5f}\t{10*np.mean(fld_errors):.5f}\n")  # cm -> mm
            f.write(f"STD [mm]:\t{10*np.std(fld_levels):.5f}\t{10*np.std(fld_errors):.5f}\n")  # cm -> mm
            f.write(f"CV: \t{100*metric.coeff_var(fld_levels):.5f}%\t{100*metric.coeff_var(fld_errors):.5f}%\n")  # Show as percentage
            # f.write(f"ICC: \t{level_icc:.5f}\t{dist_icc:.5f}\n")

            f.write(f"Reproducibility [mm]:\t{10*metric.reprod_per_field(fld_levels):.5f}\n")
            f.write(f"Stability [mm]:\t{10*metric.stab_per_field(fld_errors):.5f}\n\n")
    
    print(f"Reproducibility: {10*np.mean(total_RPDs):.5f}\n")
    print(f"Stability: {10*np.mean(total_STBs):.5f}\n\n")
    print(f"\t  Average Levels\tVertical Errors\n")
    print(f"CV: \t{100*np.mean(total_lvl_CV):.5f}\t{100*np.mean(total_dist_CV):.5f}\n")
    # print(f"ICC: \t{np.mean(total_lvl_ICC):.5f}\t{np.mean(total_dist_ICC):.5f}\n")
    print(f"Analysis on [{patient_ID}] is complete.\n\n\n\n")
    print()

    #### 3. Fraction analysis
    with open(f"{result_folder}fraction.txt", "w") as f:
        patient_path, num_fx = data.patient_path(data_root, treatment, breath)
        patient_ID = patient_path.split("/")[5].split("_")[0]
        fx_levels, fx_errors = [], []

        for fx in range(1, num_fx+1):
            fraction_path, num_fld = data.fraction_path(patient_path, fx)
            fld_levels, fld_errors = [], []

            for field in range(1, num_fld+1):
                (data_Times, data_Amps), (beam_Times, beam_Amps) = data.read_field_data(fraction_path, field)
                cutted_Amps = processing.cut_by_beams(data_Times, data_Amps, beam_Times)
                enabled_intervals, num_intvs = processing.beam_enabling_intervals(data_Times, data_Amps, beam_Times)

                for intv in range(num_intvs):
                    intv_level = metric.avg_lvl_per_interval(enabled_intervals[intv])
                    intv_error = metric.error_per_interval(enabled_intervals[intv])
                    fld_levels.append(intv_level)
                    fld_errors.append(intv_error)
                
                fx_levels.append(np.mean(fld_levels))
                fx_errors.append(np.mean(fld_errors))

        # level_data = {
        #             "Subject": list(range(1, len(fx_levels)+1)),
        #             "Rater": ["beam"] * len(fx_levels),
        #             "Score": fx_levels
        #         }
        # level_df = pd.DataFrame(level_data)
        # level_icc = pg.intraclass_corr(level_df, "Subject", "Rater", "Score")

        # dist_data = {
        #     "Subject": list(range(1, len(fx_errors)+1)),
        #     "Rater": ["beam"] * len(fx_errors),
        #     "Score": fx_errors
        # }
        # dist_df = pd.DataFrame(dist_data)
        # dist_icc = pg.intraclass_corr(dist_df, "Subject", "Rater", "Score")
        
        f.write(f"\t  Average Levels\tVertical Errors\n")
        f.write(f"Mean [mm]:\t{10*np.mean(fx_levels):.5f}\t{10*np.mean(fx_errors):.5f}\n")  # cm -> mm
        f.write(f"STD [mm]:\t{10*np.std(fx_levels):.5f}\t{10*np.std(fx_errors):.5f}\n")  # cm -> mm
        f.write(f"CV: \t{100*metric.coeff_var(fx_levels):.5f}%\t{100*metric.coeff_var(fx_errors):.5f}%\n")  # Show as percentage
        # f.write(f"ICC: \t{level_icc[['ICC']]:.5f}\t{dist_icc[['ICC']]:.5f}\n")

        f.write(f"Reproducibility [mm]:\t{10*metric.reprod_per_field(fx_levels):.5f}\n")
        f.write(f"Stability [mm]:\t{10*metric.stab_per_field(fx_errors):.5f}\n\n")
    
    print(f"Reproducibility: {10*metric.reprod_per_field(fx_levels):.5f}\n")
    print(f"Stability: {10*metric.stab_per_field(fx_errors):.5f}\n\n")
    print(f"\t  Average Levels\tVertical Errors\n")
    print(f"CV: \t{100*metric.coeff_var(fx_levels):.5f}%\t{100*metric.coeff_var(fx_errors):.5f}%\n")
    # print(level_icc[["Type", "ICC"]])
    # print(dist_icc[["Type", "ICC"]])
    
    print(f"Analysis on [{patient_ID}] is complete.")
    print()

print("Program terminated.")