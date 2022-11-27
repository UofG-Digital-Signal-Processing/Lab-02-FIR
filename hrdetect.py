import numpy as np

import constant
from firfilter import FirFilter, BandStopFilter, HighPassFilter

import matplotlib.pyplot as plt


def create_wavelet():
    t = np.arange(-0.4, 0.4, 1 / 1250)
    w = 250
    y = np.sin(w * t) / (w * t)

    return y


if __name__ == '__main__':
    y1 = create_wavelet()
    data = np.loadtxt('ECG_1000Hz.dat')
    peak_time_for_ECG = np.zeros(30000)
    peak_num = 0
    frequency_resolution = 1
    output_after_high_pass_filter = np.zeros(len(data))
    output_after_band_stop_filter = np.zeros(len(data))
    res1 = np.zeros(len(data))
    res2 = np.zeros(len(data))
    t = np.arange(0, 30000)
    t = t / 1000
    res3 = []

    high_pass_w_c = 2
    band_stop_w_c = [49, 51]
    sample_rate = 1000

    high_pass_filter = HighPassFilter(constant.sample_rate, high_pass_w_c)
    output_after_high_pass_filter = high_pass_filter.do_total_filter(data)

    band_stop_filter = BandStopFilter(constant.sample_rate, band_stop_w_c[0], band_stop_w_c[1])
    output_after_band_stop_filter = band_stop_filter.do_total_filter(output_after_high_pass_filter)

    template1 = output_after_band_stop_filter[10500:11500]  # create template
    template2 = create_wavelet()
    fir_coeff1 = template1[::-1]  # reverse time
    fir_coeff2 = template2[::-1]

    template_t = np.arange(-500, 500)
    template_t = template_t / 1000

    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.xlabel("time")
    plt.ylabel("Amplitude")
    plt.title("Template")
    plt.plot(template_t, fir_coeff1)

    plt.subplot(2, 1, 2)
    plt.xlabel("time")
    plt.ylabel("Amplitude")
    plt.title("Wavelet")
    plt.plot(template_t, fir_coeff2)
    plt.savefig("res/task_4_(Template and Wavelet).svg")

    filter1 = FirFilter(fir_coeff1)  # R-peaks filter
    filter2 = FirFilter(fir_coeff2)  # Wavelet filter

    res2 = filter2.do_total_filter(output_after_band_stop_filter)
    res2 = abs(res2)

    plt.figure(2)
    plt.xlabel("time")
    plt.ylabel("Amplitude")
    plt.title("Output with R peaks detected")
    plt.plot(t, res2)
    plt.savefig("res/task_4_(R peaks detected).svg")

    for i in range(len(data)):
        if res2[i] <= 5:
            res2[i] = 0
        else:
            peak_time_for_ECG[peak_num] = i
            peak_num += 1

    peak_time_for_ECG = peak_time_for_ECG[:peak_num]  # the place to time
    peak_time_for_ECG = peak_time_for_ECG / 1000

    res3.append(peak_time_for_ECG[0])
    for i in range(1, len(peak_time_for_ECG)):
        if peak_time_for_ECG[i] - peak_time_for_ECG[i - 1] > 0.3:  # remove the error, time interval > 0.3
            res3.append(peak_time_for_ECG[i])

    inverse_interval = []
    y_output = []
    x_output = []
    for i in range(1, len(res3)):
        inverse_interval.append((1 / (res3[i] - res3[i - 1])))  # Real heart rate
    for i in range(len(inverse_interval)):
        y_output.append(inverse_interval[i])
        y_output.append(inverse_interval[i])
    x_output.append(0)
    for i in range(1, len(y_output), 2):
        x_output.append(res3[int(i / 2)])
        x_output.append(res3[int(i / 2)])

    x_output = x_output[:-1]

    t1 = np.arange(0, 20)
    plt.figure(3)
    plt.xlabel("time")
    plt.ylabel("Amplitude")
    plt.title("Output after thresholding")
    plt.plot(t, res2)
    plt.savefig("res/task_4_(Output after thresholding).svg")

    plt.figure(4)
    plt.plot(x_output, y_output)
    plt.xlabel("time(s)")
    plt.ylabel("(/s)")
    plt.title("Momentary Heartrate")
    plt.savefig("res/task_4_(Momentary Heartrate_0).svg")
    plt.show()
