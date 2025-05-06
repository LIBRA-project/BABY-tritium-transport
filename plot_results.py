import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from festim_model_ldc import values

release_data = []
ts = []
for value in values:
    data = np.genfromtxt(
        f"testing/vel_factor_{value:.1f}/top_release.csv", delimiter=",", names=True
    )
    release_data.append(data["T_flux_surface_2"])
    ts.append(data["ts"])


fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

for t, release, value in zip(ts, release_data, values):
    axs[0].plot(t, release, label=f"vel factor {t[0]:.1f}")

    cumulative_release = cumulative_trapezoid(release, x=t, initial=0)

    axs[1].plot(t, cumulative_release, label=f"vel factor {value}", marker="o")

axs[0].set_ylabel("Tritium flux [T s-1]")
axs[1].set_ylabel("Cumulative tritium release [Bq]")

plt.xlabel("Time [s]")
plt.legend()

plt.show()
