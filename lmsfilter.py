import numpy as np

import firdesign
from firfilter import FirFilter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = 'ECG_1000Hz.dat'
    sample_rate = 1000
    frequency_resolution = 1
    noise_frequency = 50
    learning_rate = 0.001
    high_pass_w_c = 2
    data = np.loadtxt(file_path)
    n = len(data)
    # Create LMS filter
    lms_filter = FirFilter(np.zeros(int(sample_rate / frequency_resolution)))
    # Create high-pass filter
    high_pass_h = firdesign.high_pass_design(sample_rate, high_pass_w_c)
    high_pass_filter = FirFilter(high_pass_h)
    # Remove the 50Hz interference by the lms filter
    lms_output = np.zeros(n)
    for i in range(n):
        noise = np.sin(2.0 * np.pi * noise_frequency / sample_rate * i)
        lms_output[i] = lms_filter.do_filter_adaptive(data[i], noise, learning_rate)
        #print(lms_output[i])
    # Process the baseline wander by the high-pass filter
    high_pass_output = np.zeros(n)
    for i in range(n):
        high_pass_output[i] = high_pass_filter.do_filter(lms_output[i])

    # TODO Plot & Compare the results
    t = np.arange(0, n)
    t = t / sample_rate
    plt.figure(1)
    plt.title("original singal")
    plt.plot(t, data)

    plt.figure(2)
    plt.title("Filter by Adaptive LMS")
    plt.plot(t, high_pass_output)

    plt.show()