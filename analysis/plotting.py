import numpy as np
import matplotlib.pyplot as plt
from libra_toolbox.tritium.model import quantity_to_activity, ureg
from scipy.integrate import cumulative_trapezoid
import requests

s_to_day = 1 / 3600 / 24

flibe_top_inst = np.genfromtxt(
    "results/transient/instant/surface_flux_flibe_top.csv", delimiter=",", names=True
)
inconel_inner_side_inst = np.genfromtxt(
    "results/transient/instant/surface_flux_inconel_inner_side.csv",
    delimiter=",",
    names=True,
)
inconel_inner_top_inst = np.genfromtxt(
    "results/transient/instant/surface_flux_inconel_inner_top.csv",
    delimiter=",",
    names=True,
)
inconel_outer_bottom_inst = np.genfromtxt(
    "results/transient/instant/surface_flux_inconel_outer_bottom.csv",
    delimiter=",",
    names=True,
)
inconel_outer_side_inst = np.genfromtxt(
    "results/transient/instant/surface_flux_inconel_outer_side.csv",
    delimiter=",",
    names=True,
)
inconel_outer_top_inst = np.genfromtxt(
    "results/transient/instant/surface_flux_inconel_outer_top.csv",
    delimiter=",",
    names=True,
)
t_inst = flibe_top_inst["ts"]
unique_t_inst = np.unique(t_inst)
flux_flibe_top_inst = flibe_top_inst["T_flux_surface_8"]
summed_flux_flibe_top_inst = [
    flux_flibe_top_inst[t_inst == t].sum() for t in unique_t_inst
]
flux_inconel_inner_side_inst = inconel_inner_side_inst["T_flux_surface_9"]
summed_flux_inconel_inner_side_inst = [
    flux_inconel_inner_side_inst[t_inst == t].sum() for t in unique_t_inst
]
flux_inconel_inner_top_inst = inconel_inner_top_inst["T_flux_surface_10"]
summed_flux_inconel_inner_top_inst = [
    flux_inconel_inner_top_inst[t_inst == t].sum() for t in unique_t_inst
]
flux_inconel_outer_bottom_inst = inconel_outer_bottom_inst["T_flux_surface_12"]
summed_flux_inconel_outer_bottom_inst = [
    flux_inconel_outer_bottom_inst[t_inst == t].sum() for t in unique_t_inst
]
flux_inconel_outer_side_inst = inconel_outer_side_inst["T_flux_surface_13"]
summed_flux_inconel_outer_side_inst = [
    flux_inconel_outer_side_inst[t_inst == t].sum() for t in unique_t_inst
]
flux_inconel_outer_top_inst = inconel_outer_top_inst["T_flux_surface_14"]
summed_flux_inconel_outer_top_inst = [
    flux_inconel_outer_top_inst[t_inst == t].sum() for t in unique_t_inst
]
total_inner_release_inst = (
    np.array(summed_flux_flibe_top_inst)
    + np.array(summed_flux_inconel_inner_side_inst)
    + np.array(summed_flux_inconel_inner_top_inst)
) * 24
total_outer_release_inst = (
    np.array(summed_flux_inconel_outer_bottom_inst)
    + np.array(summed_flux_inconel_outer_side_inst)
    + np.array(summed_flux_inconel_outer_top_inst)
) * 24


s_to_day = 1 / 3600 / 24

inner_cumulative_release_inst = cumulative_trapezoid(
    total_inner_release_inst, x=unique_t_inst, initial=0
)
outer_cumulative_release_inst = cumulative_trapezoid(
    total_outer_release_inst, x=unique_t_inst, initial=0
)

# convert to Bq
inner_cumulative_release_inst = (
    quantity_to_activity(ureg.Quantity(inner_cumulative_release_inst, "particle"))
    .to(ureg.Bq)
    .magnitude
)
outer_cumulative_release_inst = (
    quantity_to_activity(ureg.Quantity(outer_cumulative_release_inst, "particle"))
    .to(ureg.Bq)
    .magnitude
)

# data of only recombination with local T
flibe_top_recomb = np.genfromtxt(
    "results/transient/pure_he/surface_flux_flibe_top.csv",
    delimiter=",",
    names=True,
)
inconel_inner_side_recomb = np.genfromtxt(
    "results/transient/pure_he/surface_flux_inconel_inner_side.csv",
    delimiter=",",
    names=True,
)
inconel_inner_top_recomb = np.genfromtxt(
    "results/transient/pure_he/surface_flux_inconel_inner_top.csv",
    delimiter=",",
    names=True,
)
inconel_outer_bottom_recomb = np.genfromtxt(
    "results/transient/pure_he/surface_flux_inconel_outer_bottom_calculated.csv",
    delimiter=",",
    names=True,
)
inconel_outer_side_recomb = np.genfromtxt(
    "results/transient/pure_he/surface_flux_inconel_outer_side.csv",
    delimiter=",",
    names=True,
)

inconel_outer_top_recomb = np.genfromtxt(
    "results/transient/pure_he/surface_flux_inconel_outer_top.csv",
    delimiter=",",
    names=True,
)

t_recomb = flibe_top_recomb["ts"]

unique_t_recomb = np.unique(t_recomb)

flux_flibe_top_recomb = flibe_top_recomb["T_flux_surface_8"]
summed_flux_flibe_top_recomb = [
    flux_flibe_top_recomb[t_recomb == t].sum() for t in unique_t_recomb
]
flux_inconel_inner_side_recomb = inconel_inner_side_recomb["T_flux_surface_9"]
summed_flux_inconel_inner_side_recomb = [
    flux_inconel_inner_side_recomb[t_recomb == t].sum() for t in unique_t_recomb
]
flux_inconel_inner_top_recomb = inconel_inner_top_recomb["T_flux_surface_10"]
summed_flux_inconel_inner_top_recomb = [
    flux_inconel_inner_top_recomb[t_recomb == t].sum() for t in unique_t_recomb
]
flux_inconel_outer_bottom_recomb = inconel_outer_bottom_recomb[
    "T_calculated_flux_surface_12"
]
summed_flux_inconel_outer_bottom_recomb = [
    flux_inconel_outer_bottom_recomb[t_recomb == t].sum() for t in unique_t_recomb
]
flux_inconel_outer_side_recomb = inconel_outer_side_recomb["T_flux_surface_13"]
summed_flux_inconel_outer_side_recomb = [
    flux_inconel_outer_side_recomb[t_recomb == t].sum() for t in unique_t_recomb
]
flux_inconel_outer_top_recomb = inconel_outer_top_recomb["T_flux_surface_14"]
summed_flux_inconel_outer_top_recomb = [
    flux_inconel_outer_top_recomb[t_recomb == t].sum() for t in unique_t_recomb
]

total_inner_release_recomb = (
    np.array(summed_flux_flibe_top_recomb)
    + np.array(summed_flux_inconel_inner_side_recomb)
    + np.array(summed_flux_inconel_inner_top_recomb)
) * 24
total_outer_release_recomb = (
    np.array(summed_flux_inconel_outer_bottom_recomb)
    + np.array(summed_flux_inconel_outer_side_recomb)
    + np.array(summed_flux_inconel_outer_top_recomb)
) * 24

s_to_day = 1 / 3600 / 24

inner_cumulative_release_recomb = cumulative_trapezoid(
    total_inner_release_recomb, x=unique_t_recomb, initial=0
)
outer_cumulative_release_recomb = cumulative_trapezoid(
    total_outer_release_recomb, x=unique_t_recomb, initial=0
)

# convert to Bq
inner_cumulative_release_recomb = (
    quantity_to_activity(ureg.Quantity(inner_cumulative_release_recomb, "particle"))
    .to(ureg.Bq)
    .magnitude
)
outer_cumulative_release_recomb = (
    quantity_to_activity(ureg.Quantity(outer_cumulative_release_recomb, "particle"))
    .to(ureg.Bq)
    .magnitude
)


# data of recombination with H2
flibe_top_recomb_h2 = np.genfromtxt(
    "results/transient/recomb_h2/surface_flux_flibe_top.csv", delimiter=",", names=True
)
inconel_inner_side_recomb_h2 = np.genfromtxt(
    "results/transient/recomb_h2/surface_flux_inconel_inner_side.csv",
    delimiter=",",
    names=True,
)
inconel_inner_top_recomb_h2 = np.genfromtxt(
    "results/transient/recomb_h2/surface_flux_inconel_inner_top.csv",
    delimiter=",",
    names=True,
)
inconel_outer_bottom_recomb_h2 = np.genfromtxt(
    "results/transient/recomb_h2/surface_flux_inconel_outer_bottom.csv",
    delimiter=",",
    names=True,
)
inconel_outer_side_recomb_h2 = np.genfromtxt(
    "results/transient/recomb_h2/surface_flux_inconel_outer_side.csv",
    delimiter=",",
    names=True,
)

inconel_outer_top_recomb_h2 = np.genfromtxt(
    "results/transient/recomb_h2/surface_flux_inconel_outer_top.csv",
    delimiter=",",
    names=True,
)

t_recomb_h2 = flibe_top_recomb_h2["ts"]
unique_t_recomb_h2 = np.unique(t_recomb_h2)

flux_flibe_top_recomb_h2 = flibe_top_recomb_h2["T_flux_surface_8"]
summed_flux_flibe_top_recomb_h2 = [
    flux_flibe_top_recomb_h2[t_recomb_h2 == t].sum() for t in unique_t_recomb_h2
]
flux_inconel_inner_side_recomb_h2 = inconel_inner_side_recomb_h2["T_flux_surface_9"]
summed_flux_inconel_inner_side_recomb_h2 = [
    flux_inconel_inner_side_recomb_h2[t_recomb_h2 == t].sum()
    for t in unique_t_recomb_h2
]
flux_inconel_inner_top_recomb_h2 = inconel_inner_top_recomb_h2["T_flux_surface_10"]
summed_flux_inconel_inner_top_recomb_h2 = [
    flux_inconel_inner_top_recomb_h2[t_recomb_h2 == t].sum() for t in unique_t_recomb_h2
]
flux_inconel_outer_bottom_recomb_h2 = inconel_outer_bottom_recomb_h2[
    "T_flux_surface_12"
]
summed_flux_inconel_outer_bottom_recomb_h2 = [
    flux_inconel_outer_bottom_recomb_h2[t_recomb_h2 == t].sum()
    for t in unique_t_recomb_h2
]
flux_inconel_outer_side_recomb_h2 = inconel_outer_side_recomb_h2["T_flux_surface_13"]
summed_flux_inconel_outer_side_recomb_h2 = [
    flux_inconel_outer_side_recomb_h2[t_recomb_h2 == t].sum()
    for t in unique_t_recomb_h2
]
flux_inconel_outer_top_recomb_h2 = inconel_outer_top_recomb_h2["T_flux_surface_14"]
summed_flux_inconel_outer_top_recomb_h2 = [
    flux_inconel_outer_top_recomb_h2[t_recomb_h2 == t].sum() for t in unique_t_recomb_h2
]

total_inner_release_recomb_h2 = (
    np.array(summed_flux_flibe_top_recomb_h2)
    + np.array(summed_flux_inconel_inner_side_recomb_h2)
    + np.array(summed_flux_inconel_inner_top_recomb_h2)
) * 24
total_outer_release_recomb_h2 = (
    np.array(summed_flux_inconel_outer_bottom_recomb_h2)
    + np.array(summed_flux_inconel_outer_side_recomb_h2)
    + np.array(summed_flux_inconel_outer_top_recomb_h2)
) * 24

inner_cumulative_release_recomb_h2 = cumulative_trapezoid(
    total_inner_release_recomb_h2, x=unique_t_recomb_h2, initial=0
)
outer_cumulative_release_recomb_h2 = cumulative_trapezoid(
    total_outer_release_recomb_h2, x=unique_t_recomb_h2, initial=0
)

# convert to Bq
inner_cumulative_release_recomb_h2 = (
    quantity_to_activity(ureg.Quantity(inner_cumulative_release_recomb_h2, "particle"))
    .to(ureg.Bq)
    .magnitude
)
outer_cumulative_release_recomb_h2 = (
    quantity_to_activity(ureg.Quantity(outer_cumulative_release_recomb_h2, "particle"))
    .to(ureg.Bq)
    .magnitude
)

# ##### Plotting #####


# convert to days
t_plot_recomb_h2 = unique_t_recomb_h2 * s_to_day
t_plot_recomb = unique_t_recomb * s_to_day
t_plot_instant = unique_t_inst * s_to_day

# get experimental data
# read experimental data
url = "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-1/refs/tags/v0.5/data/processed_data.json"
experimental_data = requests.get(url).json()
cumulative_release_exp_inner = experimental_data["cumulative_tritium_release"]["IV"][
    "total"
]["value"]
sampling_times_inner = experimental_data["cumulative_tritium_release"]["IV"][
    "sampling_times"
]["value"]
cumulative_release_exp_outer = experimental_data["cumulative_tritium_release"]["OV"][
    "total"
]["value"]
sampling_times_outer = experimental_data["cumulative_tritium_release"]["OV"][
    "sampling_times"
]["value"]


# plt.figure()

# plt.plot(
#     t_plot_recomb,
#     summed_flux_flibe_top_recomb,
#     label="flibe",
# )
# plt.xlabel("Time [days]")
# plt.ylabel("Surface Flux [s-1]")
# plt.legend()
# plt.xlim(left=0)
# ax = plt.gca()
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# plt.figure()

# plt.plot(
#     t_plot_recomb,
#     summed_flux_inconel_outer_side_recomb,
#     label="outer side",
# )
# plt.plot(
#     t_plot_recomb,
#     summed_flux_inconel_outer_bottom_recomb,
#     label="outer bottom",
# )
# plt.plot(
#     t_plot_recomb,
#     summed_flux_inconel_outer_top_recomb,
#     label="outer top",
# )
# # plt.plot(
# #     t_plot_instant,
# #     summed_flux_inconel_outer_side_inst,
# #     label="outer side",
# # )
# # plt.plot(
# #     t_plot_instant,
# #     summed_flux_inconel_outer_bottom_inst,
# #     label="outer bottom",
# # )
# # plt.plot(
# #     t_plot_instant,
# #     summed_flux_inconel_outer_top_inst,
# #     label="outer top",
# # )
# plt.xlabel("Time [days]")
# plt.ylabel("Surface Flux [s-1]")
# plt.legend()
# plt.xlim(left=0)
# ax = plt.gca()
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)


plt.figure()
plt.scatter(
    sampling_times_inner,
    cumulative_release_exp_inner,
    label="Experiment",
    color="#023047",
)
plt.scatter(
    sampling_times_outer,
    cumulative_release_exp_outer,
    color="tab:green",
)
# plt.plot(
#     t_plot_recomb_h2,
#     inner_cumulative_release_recomb_h2,
#     color="darkgreen",
#     label="With 1000 ppm H2",
# )
# plt.plot(
#     t_plot_recomb_h2,
#     outer_cumulative_release_recomb_h2,
#     color="green",
# )
plt.plot(
    t_plot_recomb,
    inner_cumulative_release_recomb,
    color="#023047",
)
plt.plot(
    t_plot_recomb,
    outer_cumulative_release_recomb,
    color="tab:green",
)
plt.xlabel("Time [days]")
plt.ylabel("Cumulative tritium release [Bq]")
plt.legend()
plt.ylim(bottom=0)
plt.xlim(left=0)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.show()
