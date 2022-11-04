import matplotlib.pyplot as plt

import firdesign

h = firdesign.band_stop_design(500, [45, 55])
plt.plot(h)
plt.show()
