import data, utils, processing, metric
import numpy as np

root = 'E:/Deep_Learning/Respiration/'
data_root = f"{root}DATA/"
plot_root = f"{root}PLOT/"

while True:
    proceed = int(input("Please enter '1' if you want to proceed, or enter '0': "))
    if proceed == 0: break
    treatment = input("Desired Treatment Method [Type 'STATIC' or 'ARC']: ")
    breath = input("Desired Breath Type: [Type 'Breathhold' or 'FULL']: ")

    patient_path, num_fx = data.patient_path(data_root, treatment, breath)
    for fx in range(1, num_fx+1):
        print(f"=====Fraction{fx}=====")
        fraction_path, num_fld = data.fraction_path(patient_path, fx)
        fx_reprods, fx_stabs = [], []
        for field in range(1, num_fld+1):
            print(f"=====Field{field}=====")
            data_Times, data_Amps = data.read_field_AP(fraction_path, field)
            beam_Times, beam_Amps = data.read_field_beams(fraction_path, field)
            dilated_beams = processing.dilate_beams(data_Times, beam_Times, beam_Amps)
            cutted_amps = np.array(data_Amps) * np.array(dilated_beams)
            enabled_intervals, num_intvs = processing.beam_enabling_intervals(data_Times, cutted_amps)
            field_lvls, field_errors = [], []
            for intv in range(num_intvs):
                avg_lvl = metric.avg_lvl_per_interval(enabled_intervals[intv])
                error = metric.error_per_interval(enabled_intervals[intv])
                field_lvls.append(avg_lvl)
                field_errors.append(error)
                print(f"[Interval{intv}] Average Level: {avg_lvl}\tVertical Error: {error}")
            reprod = metric.reprod_per_field(field_lvls)
            stab = metric.stab_per_field(field_errors)
            fx_reprods.append(reprod)
            fx_stabs.append(stab)
            print(f"Reproducibility: {reprod}\tStability: {stab}")
        mean_reprod = metric.mean_reprod_per_fraction(fx_reprods)
        mean_stab = metric.mean_stab_per_fraction(fx_stabs)
        print(f"Mean Reproducibility: {mean_reprod}\tMean Stability: {mean_stab}")

print("Program terminated.")