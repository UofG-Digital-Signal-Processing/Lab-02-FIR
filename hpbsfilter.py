import numpy as np

import firdesign
from firfilter import FirFilter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = 'ECG_1000Hz.dat'
    sample_rate = 1000
    band_stop_w_1 = 49
    band_stop_w_2 = 51
    high_pass_w_c = 2
    data = np.loadtxt(file_path)
    n = len(data)
    # Create bands-top filter
    band_stop_h = firdesign.band_stop_design(sample_rate, [band_stop_w_1, band_stop_w_2])
    band_stop_filter = FirFilter(band_stop_h)
    # Create high-pass filter
    high_pass_h = firdesign.high_pass_design(sample_rate, high_pass_w_c)
    high_pass_filter = FirFilter(high_pass_h)
    # Remove the 50Hz interference by the band-stop filter
    band_stop_output = np.zeros(n)
    for i in range(n):
        band_stop_output[i] = band_stop_filter.do_filter(data[i])
    # Process the baseline wander by the high-pass filter
    high_pass_output = np.zeros(n)
    for i in range(n):
        high_pass_output[i] = high_pass_filter.do_filter(band_stop_output[i])

    # TODO Plot & Compare the results

    # x1 = range(0,n)
    # plt.figure(1)
    # plt.subplot(2, 1, 1)
    # plt.xlabel("time")
    # plt.ylabel("Amplitude")
    # plt.title("original_ECG")
    # plt.plot(x1, data)
    #
    # plt.show()

    amplitude = np.array(band_stop_output)

    # calculating the total number of samples
    total_samples = np.size(band_stop_output)

    # calculating the time step between each sample
    time_step = 1 / sample_rate

    # calculating the time domain for the signal
    time_domain = np.linspace(0, (total_samples - 1) * time_step, total_samples)

    # calculating the frequency step size for the signal
    freq_step = sample_rate / total_samples

    # calculating the frequency domain for the signal
    freq_domain = np.linspace(0, (total_samples - 1) * freq_step, total_samples)
    freq_domain_plt = freq_domain[:int(total_samples / 2) + 1]

    # calculating the frequency response of the signal
    freq_mag = np.fft.fft(band_stop_output)
    freq_mag_abs = np.abs(freq_mag) / total_samples
    freq_mag_abs_plt = 2 * freq_mag_abs[:int(total_samples / 2) + 1]
    freq_mag_dB = 20 * np.log10(freq_mag_abs_plt)

    plt.figure(4)
    plt.title("Spectrum of output after eliminating 50Hz")
    plt.xlabel("frequency")
    plt.ylabel("dB")
    plt.plot(freq_domain_plt, freq_mag_abs_plt)

    t = np.arange(0, 30000)
    t = t / 250
    plt.figure(1)
    plt.title("original signal")
    plt.plot(t, data)

    plt.figure(2)
    plt.title("output after eliminating baseline wander")
    plt.plot(t, high_pass_output)

    plt.figure(3)
    plt.title("output after eliminating 50Hz")
    plt.plot(t, band_stop_output)
    plt.show()

