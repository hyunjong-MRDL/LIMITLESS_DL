import data, processing, metric, plot
import numpy as np

root = 'E:/Deep_Learning/Respiration/'
data_root = f"{root}DATA/"
plot_root = f"{root}PLOT/"
num_attempts = 2 # MTG_{number}
plot_mode = 0 # 0: Off, 1: On

while True:
    proceed = int(input("Please enter '1' if you want to proceed, or enter '0': "))
    if proceed == 0: break
    treatment = input("Desired Treatment Method [Type 'STATIC' or 'ARC']: ")
    breath = input("Desired Breath Type: [Type 'Breathhold' or 'FULL']: ")

    patient_path, num_fx = data.patient_path(data_root, treatment, breath)
    patient_ID = patient_path.split("/")[5].split("_")[0]
    plot_folder = f"{plot_root}/MTG_{num_attempts}/{patient_ID}/"
    with open(f"{root}sample.txt", "w") as f:
        total_reprod, total_stab = [], []
        for fx in range(1, num_fx+1):
            f.write(f"\t\t\t\t=====Fraction{fx}=====\n\n")
            fraction_path, num_fld = data.fraction_path(patient_path, fx)
            fx_reprods, fx_stabs = [], []
            for field in range(1, num_fld+1):
                f.write(f"\t\t\t\t =====Field{field}=====\n")
                data_Times, data_Amps = data.read_field_AP(fraction_path, field)
                beam_Times, beam_Amps = data.read_field_beams(fraction_path, field)
                dilated_beams = processing.dilate_beams(data_Times, beam_Times, beam_Amps)
                cutted_amps = np.array(data_Amps) * np.array(dilated_beams)
                if plot_mode == 1:
                    plot.plot_AP(plot_folder, fx, field, "Raw", data_Times, data_Amps)
                    plot.plot_AP(plot_folder, fx, field, "Cutted", data_Times, cutted_amps)
                enabled_intervals, num_intvs = processing.beam_enabling_intervals(data_Times, cutted_amps)
                field_lvls, field_errors = [], []
                for intv in range(num_intvs):
                    avg_lvl = metric.avg_lvl_per_interval(enabled_intervals[intv])
                    error = metric.error_per_interval(enabled_intervals[intv])
                    field_lvls.append(avg_lvl)
                    field_errors.append(error)
                    f.write(f"[Interval{intv}] Average Level: {avg_lvl}\tVertical Error: {error}\n")
                reprod = metric.reprod_per_field(field_lvls)
                stab = metric.stab_per_field(field_errors)
                fx_reprods.append(reprod)
                fx_stabs.append(stab)
                f.write(f"Reproducibility: {reprod}\tStability: {stab}\n\n")
            mean_reprod = metric.mean_reprod_per_fraction(fx_reprods)
            mean_stab = metric.mean_stab_per_fraction(fx_stabs)
            total_reprod.append(mean_reprod)
            total_stab.append(mean_stab)
            f.write(f"Mean Reproducibility: {mean_reprod}\tMean Stability: {mean_stab}\n\n\n")
    print(f"Analysis on [{patient_ID}] is complete.")

print("Program terminated.")