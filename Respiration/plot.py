import os, utils
from scipy.fft import fft, fftshift
import numpy as np
import matplotlib.pyplot as plt

def plot_AP(plot_folder, fx, field, plot_type, data_Times, data_Amps):
    plot_folder = f"{plot_folder}AP/{fx}/"
    utils.createFolder(plot_folder)
    
    filename = f"{plot_folder}{fx}fx_field{field} ({plot_type}).jpg"

    if os.path.exists(filename):
        print(f"AP Plot [{fx}fx_field{field} ({plot_type}).jpg] already exists.")
    else:
        plt.figure(figsize=(18, 12))
        plt.plot(data_Times, data_Amps)
        plt.title(f"{fx}fx_field{field} (AP)")
        plt.xlabel("Time (s)"), plt.ylabel("Amplitude (cm)")
        plt.savefig(filename)
        print(f"AP Plot [{fx}fx_field{field} ({plot_type}).jpg] saved successfully.")
        plt.close()

def plot_FFT(plot_folder, fx, field, plot_type, data_Times, data_Amps):
    plot_folder = f"{plot_folder}FFT/{fx}/"
    utils.createFolder(plot_folder)

    filename = f"{plot_folder}{fx}fx_field{field} ({plot_type}).jpg"
    
    if os.path.exists(filename):
        print(f"FFT Plot [{fx}fx_field{field}({plot_type}).jpg] already exists.")
    else:
        fft_amp = fftshift(fft(np.array(data_Amps)))
        N = len(data_Amps)
        fs = 1 / (data_Times[1] - data_Times[0])
        freqs = np.linspace(0, N-1, N) * (fs/N)
        plt.figure(figsize=(18, 12))
        plt.plot(freqs, np.abs(fft_amp))
        plt.title(f"{fx}fx_field{field} (FFT)")
        plt.xlabel("Frequency (Hz)"), plt.ylabel("Amplitude (cm)")
        plt.savefig(filename)
        print(f"FFT Plot [{fx}fx_field{field} ({plot_type}).jpg] saved successfully.")
        plt.close()