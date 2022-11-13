import matplotlib.pyplot as plt
import numpy as np

import constant
import util
from firfilter import LmsFilter, HighPassFilter

import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':
    noise_freq = 50
    learning_rate = 0.001
    high_pass_w_c = 2
    original_data = np.loadtxt(constant.file_path)
    n = len(original_data)
    # Create LMS filter
    lms_filter = LmsFilter(constant.sample_rate, noise_freq, learning_rate, constant.freq_resolution)
    # Create high-pass filter
    high_pass_filter = HighPassFilter(constant.sample_rate, high_pass_w_c)
    # Remove the 50Hz interference by the lms filter
    filtered_data = lms_filter.do_total_filter(original_data)
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
    plt.savefig("res/task_3.svg")
    plt.show()
