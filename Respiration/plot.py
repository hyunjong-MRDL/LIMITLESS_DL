import os, utils, processing
from scipy.fft import fft, fftshift
import numpy as np
import matplotlib.pyplot as plt

"""Plot Anterior-Posterior data"""
def plot_AP(result_folder, fx, field, plot_type, data_Times, data_Amps, savefig):
    result_folder = f"{result_folder}AP/{fx}/"
    utils.createFolder(result_folder)
    
    filename = f"{result_folder}{fx}fx_field{field} ({plot_type}).jpg"

    plt.figure(figsize=(18, 12))
    plt.plot(data_Times, data_Amps)
    plt.title(f"{fx}fx_field{field} (AP)")
    plt.xlabel("Time (s)"), plt.ylabel("Amplitude (cm)")

    if savefig == True:
        if os.path.exists(filename):
            print(f"AP Plot [{fx}fx_field{field} ({plot_type}).jpg] already exists.")
        else:
            plt.savefig(filename)
            print(f"AP Plot [{fx}fx_field{field} ({plot_type}).jpg] saved successfully.")

    plt.close()

"""Plot [Cutted Amps] and [Dilated metrics] in a same figure"""
def integrated_plot(result_folder, fx, field, data_Times, cutted_Amps, dilated_avgs, dilated_lines, savefig):
    result_folder = f"{result_folder}AP/{fx}/"
    utils.createFolder(result_folder)
    
    filename = f"{result_folder}{fx}fx_field{field} (Integrated).jpg"

    plt.figure(figsize=(18, 12))
    plt.plot(data_Times, cutted_Amps), plt.plot(data_Times, dilated_avgs), plt.plot(data_Times, dilated_lines)
    plt.title(f"{fx}fx_field{field} (Integrated)")
    plt.xlabel("Time (s)"), plt.ylabel("Amplitude (cm)")

    if savefig == True:
        if os.path.exists(filename):
            print(f"AP Plot [{fx}fx_field{field} (Integrated).jpg] already exists.")
        else:
            plt.savefig(filename)
            print(f"AP Plot [{fx}fx_field{field} (Integrated).jpg] saved successfully.")

    plt.close()

def metric_plot(result_folder, total_metrics, metric_type, savefig):
    result_folder = f"{result_folder}Metric_Plot/"
    utils.createFolder(result_folder)
    
    filename = f"{result_folder}{metric_type}.jpg"

    plt.figure(figsize=(18, 12))
    plt.plot(range(len(total_metrics)), total_metrics)
    plt.title(f"{metric_type}")
    plt.xlabel("Fraction"), plt.ylabel("Amplitude (cm)")

    if savefig == True:
        if os.path.exists(filename):
            print(f"Metric Plot [{metric_type}] already exists.")
        else:
            plt.savefig(filename)
            print(f"Metric Plot [{metric_type}] saved successfully.")

    plt.close()

"""Plot FFT of the AP data"""
def plot_FFT(result_folder, fx, field, plot_type, data_Times, data_Amps, savefig):
    result_folder = f"{result_folder}FFT/{fx}/"
    utils.createFolder(result_folder)

    filename = f"{result_folder}{fx}fx_field{field} ({plot_type}).jpg"

    fft_amp = fftshift(fft(np.array(data_Amps)))
    N = len(data_Amps)
    fs = 1 / (data_Times[1] - data_Times[0])
    freqs = np.linspace(0, N-1, N) * (fs/N)
    plt.figure(figsize=(18, 12))
    plt.plot(freqs, np.abs(fft_amp))
    plt.title(f"{fx}fx_field{field} (FFT)")
    plt.xlabel("Frequency (Hz)"), plt.ylabel("Amplitude (cm)")
    
    if savefig == True:
        if os.path.exists(filename):
            print(f"FFT Plot [{fx}fx_field{field}({plot_type}).jpg] already exists.")
        else:
            plt.savefig(filename)
            print(f"FFT Plot [{fx}fx_field{field} ({plot_type}).jpg] saved successfully.")
    plt.close()