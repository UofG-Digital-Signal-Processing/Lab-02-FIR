import matplotlib.pyplot as plt
import numpy as np

import firdesign
import util
from firfilter import FirFilter

if __name__ == '__main__':
    file_path = 'ECG_1000Hz.dat'
    sample_rate = 1000
    band_stop_w_1 = 49
    band_stop_w_2 = 51
    high_pass_w_c = 2
    original_data = np.loadtxt(file_path)
    n = len(original_data)
    # Create bands-top filter
    band_stop_h = firdesign.band_stop_design(sample_rate, [band_stop_w_1, band_stop_w_2])
    band_stop_filter = FirFilter(band_stop_h)
    # Create high-pass filter
    high_pass_h = firdesign.high_pass_design(sample_rate, high_pass_w_c)
    high_pass_filter = FirFilter(high_pass_h)
    # Remove the 50Hz interference by the band-stop filter
    band_stop_output = np.zeros(n)
    for i in range(n):
        band_stop_output[i] = band_stop_filter.do_filter(original_data[i])
    # Process the baseline wander by the high-pass filter
    high_pass_output = np.zeros(n)
    for i in range(n):
        high_pass_output[i] = high_pass_filter.do_filter(band_stop_output[i])

    # TODO Plot & Compare the results
    # Plot the original ECG time domain
    original_time, original_amplitude = util.cal_time_domain(original_data, sample_rate)
    plt.subplot(2, 2, 1)
    plt.plot(original_time, original_amplitude)
    plt.title('Original ECG Time Domain')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    # Plot the original ECG frequency domain
    original_freq, original_amplitude = util.cal_frequency_domain(original_data, sample_rate)
    plt.subplot(2, 2, 2)
    plt.plot(original_freq, original_amplitude)
    plt.title('Original ECG Frequency Domain')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    # Plot the filtered ECG time domain
    filtered_time, filtered_amplitude = util.cal_time_domain(high_pass_output, sample_rate)
    plt.subplot(2, 2, 3)
    plt.plot(filtered_time, filtered_amplitude)
    plt.title('Filtered ECG Time Domain')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    # Plot the filtered ECG frequency domain
    filtered_freq, filtered_amplitude = util.cal_frequency_domain(high_pass_output, sample_rate)
    plt.subplot(2, 2, 4)
    plt.plot(filtered_freq, filtered_amplitude)
    plt.title('Filtered ECG Frequency Domain')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("res/task_2.svg")
    plt.show()
