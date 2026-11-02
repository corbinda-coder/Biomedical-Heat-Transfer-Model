
"""
Biomedical Transport Phenomena
Laser irradiation + cryogen surface cooling

1D transient finite-difference heat-transfer model.
Temperature is calculated as a function of time and tissue depth.
"""

import numpy as np


# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

# Tissue properties
RHO = 1090.0             # Density (kg/m^3)
CP = 3500.0              # Specific heat (J/kg/K)
K = 0.41                 # Thermal conductivity (W/m/K)

ALPHA = K / (RHO * CP)   # Thermal diffusivity (m^2/s)

# Initial tissue temperature
T_INITIAL = 37.0         # Celsius

# Laser parameters
LASER_WAVELENGTH = 595e-9   # Wavelength (m)
FLUENCE = 8.0e4             # 8 J/cm^2 converted to J/m^2
PULSE_DURATION = 0.002     # 2 ms (s)
ABSORBED_FRACTION = 0.15   # Assumed absorbed fraction

# Laser energy deposition profile
TARGET_DEPTH = 0.0002     # 0.20 mm (m)
SOURCE_WIDTH = 0.00005    # 0.05 mm (m)

# Cryogen cooling parameters
CRYOGEN_TEMP = -26.2      # Celsius
H = 1500.0                # Heat-transfer coefficient (W/m^2/K)
SPRAY_DURATION = 0.030    # 30 ms (s)

# Tissue computational domain
TISSUE_DEPTH = 0.005      # 5 mm (m)

# Simulation duration
SIMULATION_TIME = 0.120   # 120 ms (s)

# Numerical resolution
DZ = 25e-6                # 25 micrometers (m)
DT = 0.0005               # 0.5 ms (s)


# ============================================================
# 2. LASER HEAT SOURCE
# ============================================================

def laser_heat_source(z):
    """
    Gaussian volumetric laser energy distribution.

    The source is centered at the assumed vessel depth.
    Its integral over depth is normalized to 1.
    """

    profile = np.exp(
        -0.5 * ((z - TARGET_DEPTH) / SOURCE_WIDTH) ** 2
    )

    # Normalize so integral(profile dz) = 1
    profile /= np.trapezoid(profile, z)

    return profile


# ============================================================
# 3. NUMERICAL MODEL
# ============================================================

def simulate():

    # Spatial and temporal grids
    z = np.arange(0, TISSUE_DEPTH + DZ / 2, DZ)
    time = np.arange(
        0, SIMULATION_TIME + DT / 2, DT
    )

    nodes = len(z)
    steps = len(time)

    # Stability parameters
    r = ALPHA * DT / DZ**2
    beta = 2 * H * DT / (RHO * CP * DZ)

    if r > 0.5:
        raise ValueError("FTCS interior stability condition violated.")

    if 2 * r + beta > 1:
        raise ValueError(
            "Time step may be unstable at the cooled surface. "
            "Reduce DT."
        )

    # Initialize temperature matrix
    # Rows = time, columns = tissue depth
    T = np.full((steps, nodes), T_INITIAL)

    # Normalized laser absorption distribution
    g = laser_heat_source(z)

    # Laser energy deposited per unit area
    absorbed_energy = ABSORBED_FRACTION * FLUENCE

    # ========================================================
    # TIME INTEGRATION
    # ========================================================

    for n in range(steps - 1):

        current = T[n]
        updated = current.copy()

        # Laser heat source (W/m^3)
        if time[n] < PULSE_DURATION:

            Q = (
                absorbed_energy
                / PULSE_DURATION
                * g
            )

        else:
            Q = np.zeros(nodes)

        # ----------------------------------------------------
        # Interior tissue nodes
        # ----------------------------------------------------

        updated[1:-1] = (
            current[1:-1]
            + r * (
                current[2:]
                - 2 * current[1:-1]
                + current[:-2]
            )
            + DT * Q[1:-1] / (RHO * CP)
        )

        # ----------------------------------------------------
        # Surface boundary: cryogen convection
        # ----------------------------------------------------

        surface_cooling = 0.0

        if time[n] < SPRAY_DURATION:

            surface_cooling = (
                2 * H
                * (CRYOGEN_TEMP - current[0])
                / (RHO * CP * DZ)
            )

        updated[0] = (
            current[0]
            + 2 * r * (current[1] - current[0])
            + DT * surface_cooling
            + DT * Q[0] / (RHO * CP)
        )

        # ----------------------------------------------------
        # Bottom boundary: insulated (zero heat flux)
        # ----------------------------------------------------

        updated[-1] = (
            current[-1]
            + 2 * r * (current[-2] - current[-1])
            + DT * Q[-1] / (RHO * CP)
        )

        T[n + 1] = updated

    return z, time, T


# ============================================================
# 4. TEMPERATURE ANALYSIS
# ============================================================

def analyze_results(z, time, T):

    print("\n========== SIMULATION RESULTS ==========")

    depths = [
        0.0,
        0.00005,
        0.0002,
        0.0005
    ]

    for depth in depths:

        index = np.argmin(np.abs(z - depth))

        peak_index = np.argmax(T[:, index])

        peak_temp = T[peak_index, index]
        peak_time = time[peak_index]

        print(
            f"Depth: {depth * 1000:.2f} mm | "
            f"Peak temperature: {peak_temp:.2f} °C | "
            f"Peak time: {peak_time * 1000:.2f} ms"
        )

    print(
        f"\nMinimum surface temperature: "
        f"{np.min(T[:, 0]):.2f} °C"
    )

    target_index = np.argmin(
        np.abs(z - TARGET_DEPTH)
    )

    print(
        f"Maximum target temperature: "
        f"{np.max(T[:, target_index]):.2f} °C"
    )

    print("========================================\n")


# ============================================================
# 5. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":

    z, time, T = simulate()

    analyze_results(z, time, T)
