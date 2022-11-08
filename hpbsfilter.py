import matplotlib.pyplot as plt
import numpy as np

import constant
import util
from firfilter import BandStopFilter, HighPassFilter

if __name__ == '__main__':
    band_stop_w_1 = 49
    band_stop_w_2 = 51
    high_pass_w_c = 2
    original_data = np.loadtxt(constant.file_path)
    n = len(original_data)
    # Create band-stop filter
    band_stop_filter = BandStopFilter(constant.sample_rate, band_stop_w_1, band_stop_w_2)
    # Create high-pass filter
    high_pass_filter = HighPassFilter(constant.sample_rate, high_pass_w_c)
    # Remove the 50Hz interference by the band-stop filter
    filtered_data = band_stop_filter.do_total_filter(original_data)
    # Process the baseline wander by the high-pass filter
    filtered_data = high_pass_filter.do_total_filter(filtered_data)
    # Plot the original ECG time domain
    original_time, original_amplitude = util.cal_time_domain(original_data, constant.sample_rate)
    plt.subplot(2, 2, 1)
    plt.plot(original_time, original_amplitude)
    plt.title('Original ECG Time Domain')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    # Plot the original ECG frequency domain
    original_freq, original_amplitude = util.cal_frequency_domain(original_data, constant.sample_rate)
    plt.subplot(2, 2, 2)
    plt.plot(original_freq, original_amplitude)
    plt.title('Original ECG Frequency Domain')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    # Plot the filtered ECG time domain
    filtered_time, filtered_amplitude = util.cal_time_domain(filtered_data, constant.sample_rate)
    plt.subplot(2, 2, 3)
    plt.plot(filtered_time, filtered_amplitude)
    plt.title('Filtered ECG Time Domain')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    # Plot the filtered ECG frequency domain
    filtered_freq, filtered_amplitude = util.cal_frequency_domain(filtered_data, constant.sample_rate)
    plt.subplot(2, 2, 4)
    plt.plot(filtered_freq, filtered_amplitude)
    plt.title('Filtered ECG Frequency Domain')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("res/task_2.svg")
    plt.show()
