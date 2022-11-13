 # TODO Detect R-peaks in ECG signal
import matplotlib
import numpy as np
import constant

import firdesign
from firfilter import FirFilter, BandStopFilter,HighPassFilter

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def createWavelet():
    t = np.arange(-0.4, 0.4, 1 / 1250)
    w = 250
    y = np.sin(w * t) / (w * t)

    return y


if __name__ == '__main__':
    y1 = createWavelet()
    data = np.loadtxt('ECG_1000Hz.dat')
    PeakTimeForECG = np.zeros(30000)
    # data_max = np.min(data)
    # data = data/data_max
    # plt.plot(data)
    # plt.show()
    PeakNum = 0
    Frequency_Resolution = 1
    OutputAfterHighpassFilter = np.zeros(len(data))
    OutputAfterBandStopFilter = np.zeros(len(data))
    res1 = np.zeros(len(data))
    res2 = np.zeros(len(data))
    t = np.arange(0, 30000)
    t = t / 1000
    res3 = []

    high_pass_w_c = 2
    band_stop_w_c = [49, 51]
    sample_rate = 1000

    high_pass_filter = HighPassFilter(constant.sample_rate,high_pass_w_c)
    OutputAfterHighpassFilter = high_pass_filter.do_total_filter(data)

    band_stop_filter = BandStopFilter(constant.sample_rate, band_stop_w_c[0], band_stop_w_c[1])
    OutputAfterBandStopFilter = band_stop_filter.do_total_filter(OutputAfterHighpassFilter)

    template1 = OutputAfterBandStopFilter[10500:11500]  # create template
    template2 = createWavelet()
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


    filter1 = FirFilter(fir_coeff1) # R-peaks filter
    filter2 = FirFilter(fir_coeff2) # Wavelet filter

    res2 = filter2.do_total_filter(OutputAfterBandStopFilter)
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
            PeakTimeForECG[PeakNum] = i
            PeakNum += 1

    PeakTimeForECG = PeakTimeForECG[:PeakNum] # the place to time
    PeakTimeForECG = PeakTimeForECG / 1000

    res3.append(PeakTimeForECG[0])
    for i in range(1, len(PeakTimeForECG)):
        if PeakTimeForECG[i] - PeakTimeForECG[i - 1] > 0.3: # remove the error, time interval > 0.3
            res3.append(PeakTimeForECG[i])

    InverseInterval = []
    y_output = []
    x_output = []
    for i in range(1, len(res3)):
        InverseInterval.append((1 / (res3[i] - res3[i - 1]))) # Real heart rate
    # FirstValue = InverseInterval[0]
    # InverseInterval.insert(0,FirstValue)
    for i in range(len(InverseInterval)):
        y_output.append(InverseInterval[i])
        y_output.append(InverseInterval[i])
    x_output.append(0)
    for i in range(1, len(y_output),2):
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

