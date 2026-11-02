
"""
Run the Biomedical Transport Phenomena heat-transfer model.

Generates:
1. Temperature vs. time at selected tissue depths.
2. Temperature vs. depth at selected times.
"""

import numpy as np
import matplotlib.pyplot as plt

from tissue_heat_model import simulate, analyze_results


# ============================================================
# 1. RUN SIMULATION
# ============================================================

print("Running tissue heat-transfer simulation...")

z, time, T = simulate()

print("Simulation complete.")

analyze_results(z, time, T)


# ============================================================
# 2. TEMPERATURE VS TIME
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

selected_depths = [
    (0.0, "Skin surface (0 mm)"),
    (0.00005, "Epidermis (0.05 mm)"),
    (0.0002, "Vessel target (0.20 mm)"),
    (0.0005, "Dermis (0.50 mm)")
]

for depth, label in selected_depths:

    index = np.argmin(np.abs(z - depth))

    ax.plot(
        time * 1000,
        T[:, index],
        linewidth=2,
        label=label
    )

# Highlight laser and cryogen application periods
ax.axvspan(
    0, 2,
    alpha=0.12,
    label="Laser pulse (2 ms)"
)

ax.axvspan(
    0, 30,
    alpha=0.12,
    label="Cryogen spray (30 ms)"
)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Temperature (°C)")

ax.set_title(
    "Tissue Temperature vs. Time\n"
    "Laser Irradiation + Cryogen Cooling"
)

ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()

fig.savefig(
    "temperature_vs_time.png",
    dpi=300
)


# ============================================================
# 3. TEMPERATURE VS DEPTH
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

selected_times = [
    0.002,
    0.030,
    0.120
]

for selected_time in selected_times:

    index = np.argmin(
        np.abs(time - selected_time)
    )

    ax.plot(
        z * 1000,
        T[index, :],
        linewidth=2,
        label=f"t = {time[index] * 1000:.0f} ms"
    )

# Highlight vessel target depth
ax.axvline(
    0.20,
    linestyle="--",
    linewidth=1.5,
    label="Vessel target (0.20 mm)"
)

ax.set_xlabel("Tissue Depth (mm)")
ax.set_ylabel("Temperature (°C)")

ax.set_title(
    "Tissue Temperature vs. Depth\n"
    "At Selected Times"
)

ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()

fig.savefig(
    "temperature_vs_depth.png",
    dpi=300
)


# ============================================================
# 4. SHOW GRAPHS
# ============================================================

plt.show()

print("Graphs saved successfully.")
