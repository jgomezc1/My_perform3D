# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:14:59 2025

@author: jgomez
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:14:59 2025

@author: jgomez
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
from scipy.fft import fft, ifft

def plot_displacement_history(filename, dof, ymin=None, ymax=None):
    """
    Plots the displacement history for a given DOF from an OpenSees node recorder output.

    Parameters:
    filename (str): Path to the data file.
    dof (int): Degree of freedom to plot (1 = Ux, 2 = Uy, 3 = Uz/ThetaZ).
    ymin (float, optional): Minimum y-axis value. If None, auto-scaled symmetrically.
    ymax (float, optional): Maximum y-axis value. If None, auto-scaled symmetrically.
    """
    # Load the data
    data = pd.read_csv(filename, delim_whitespace=True, header=None)
    
    # Validate DOF
    if dof not in [1, 2, 3]:
        raise ValueError("Degree of freedom must be 1 (Ux), 2 (Uy), or 3 (Uz/ThetaZ).")
    
    # Extract time and selected displacement
    time = data[0]
    displacement = data[dof]
    
    # Auto-scale y-axis if not provided
    if ymin is None or ymax is None:
        max_val = max(abs(displacement.max()), abs(displacement.min()))
        ymin = -max_val
        ymax = max_val
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time, displacement, label=f'DOF {dof}')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.title(f'Displacement History for DOF {dof}')
    plt.ylim([ymin, ymax])
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_base_shear_vs_displacement(disp_data, force_data, direction='X', 
                                     xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Plots base shear vs master node displacement.

    Parameters:
    - disp_data: DataFrame from node displacement history.
    - force_data: DataFrame from column forces history.
    - direction: 'X' for Fx vs Ux, 'Y' for Fy vs Uy.
    - xmin, xmax, ymin, ymax: Optional axis limits. If None, symmetric auto-scaling is used.
    """
    time = disp_data[0]
    ux = disp_data[1]
    uy = disp_data[2]

    # Base shear from all 4 columns
    base_shear_y = force_data[3] + force_data[15] + force_data[27] + force_data[39]
    base_shear_x = force_data[1] + force_data[13] + force_data[25] + force_data[37]

    # Select based on direction
    if direction.upper() == 'X':
        x_data = ux
        y_data = base_shear_x
        x_label = 'Ux [m]'
        y_label = 'Base Shear Fx [N]'
    else:
        x_data = uy
        y_data = base_shear_y
        x_label = 'Uy [m]'
        y_label = 'Base Shear Fy [N]'

    # Axis scaling
    if xmin is None or xmax is None:
        xmax = max(abs(x_data.max()), abs(x_data.min()))
        xmin = -xmax
    if ymin is None or ymax is None:
        ymax = max(abs(y_data.max()), abs(y_data.min()))
        ymin = -ymax

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, label=f'{y_label} vs {x_label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label}')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
def export_dof_displacement(data, dof, output_filename): 
    """
    Extracts displacement for a specified DOF (1=Ux, 2=Uy, 3=ThetaZ) and writes it to a text file 
    without time or headers.

    Parameters:
    - data (pd.DataFrame): DataFrame with columns [time, Ux, Uy, ThetaZ or Uz]
    - dof (int): Degree of freedom to extract (1 = Ux, 2 = Uy, 3 = ThetaZ)
    - output_filename (str): Output file name (e.g., 'Ux_only.txt')
    """
    if dof not in [1, 2, 3]:
        raise ValueError("Supported DOFs are 1 (Ux), 2 (Uy), and 3 (ThetaZ).")

    # Extract only the displacement column for the specified DOF
    displacement = data[dof]

    # Write to file with no header or index
    displacement.to_csv(output_filename, index=False, header=False, float_format="%.8f")
    
    
def load_pushover_phases(filename, dt=1.0, plot=True, step_ratio=0.05):
    """
    Reads a file containing max displacement values (maxU),
    determines safe step sizes (dU) per phase, and returns
    a list of pushover phases for displacement control analysis.

    Parameters:
        filename (str): Path to the file with one maxU value per line.
        dt (float): Time step between phase points (for plotting only).
        plot (bool): Whether to plot the displacement phases vs time.
        step_ratio (float): Fraction of maxU to use as base step (e.g., 0.05 = 5%).

    Returns:
        tuple: (phases: list of dicts, dU_list: list of float)
    """
    def get_condition_fn(maxU):
        return (lambda u, maxU=maxU: u < maxU) if maxU > 0 else (lambda u, maxU=maxU: u > maxU)

    phases = []
    maxU_list = []
    dU_list = []

    with open(filename, 'r') as file:
        for line in file:
            try:
                maxU = float(line.strip())
                if maxU == 0.0:
                    continue  # Skip zero displacement
                dU_magnitude = abs(maxU) * step_ratio
                dU = dU_magnitude if maxU > 0 else -dU_magnitude
                condition = get_condition_fn(maxU)
                phases.append({"dU": dU, "maxU": maxU, "condition": condition})
                maxU_list.append(maxU)
                dU_list.append(dU)
            except ValueError:
                continue  # Skip lines that can't be parsed

    if plot and maxU_list:
        time = np.arange(len(maxU_list)) * dt
        plt.figure(figsize=(8, 5))
        plt.plot(time, maxU_list, linewidth=2)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Target Displacement [m]')
        plt.title('Pushover Phases vs Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return phases, dU_list

def plot_displacement_vs_time(displacement_data, dt):
    """
    Plots displacement vs. time for a given displacement time history.

    Parameters:
        displacement_data (np.ndarray): 1D array of displacement values.
        dt (float): Time step between data points.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    time = np.arange(len(displacement_data)) * dt

    plt.figure(figsize=(8, 5))
    plt.plot(time, displacement_data, linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.title('Displacement vs Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_ux100_vs_time(ux100_history, dt):
    """
    Plots Ux displacement at node 4 vs time.

    Parameters:
        ux4_history (list or np.ndarray): Displacement history of node 4.
        dt (float): Time step used during each `analyze(1)` call.
    """
    ux100_array = np.array(ux100_history)
    time = np.arange(len(ux100_array)) * dt

    plt.figure(figsize=(8, 5))
    plt.plot(time, ux100_array, linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement at Node 4 (Ux) [m]')
    plt.title('Ux at Node 100 vs Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def double_integrate_acceleration(file_path: str, dt: float, plot: bool = False, 
                                  velocity_outfile: str = "velocity.out", 
                                  displacement_outfile: str = "displacement.out"):
    """
    Reads a relative acceleration time series from a file and performs numerical integration
    twice to obtain velocity and displacement time series. Optionally plots and saves results.

    Parameters:
        file_path (str): Path to the file containing acceleration data (one or two columns).
        dt (float): Time step between samples.
        plot (bool): If True, generates plots for velocity and displacement.
        velocity_outfile (str): Filename to save the velocity data (without time).
        displacement_outfile (str): Filename to save the displacement data (without time).

    Returns:
        time (np.ndarray): Time vector.
        velocity (np.ndarray): Velocity time series (m/s).
        displacement (np.ndarray): Displacement time series (m).
    """
    accel_data = np.loadtxt(file_path)

    if accel_data.ndim == 2 and accel_data.shape[1] == 2:
        accel = accel_data[:, 1]
    else:
        accel = accel_data

    time = np.arange(0, len(accel)) * dt
    velocity = cumtrapz(accel, dx=dt, initial=0.0)
    displacement = cumtrapz(velocity, dx=dt, initial=0.0)

    # Save results to files (without time)
    np.savetxt(velocity_outfile, velocity, fmt='%.6e')
    np.savetxt(displacement_outfile, displacement, fmt='%.6e')
    print(f"✅ Velocity saved to '{velocity_outfile}'")
    print(f"✅ Displacement saved to '{displacement_outfile}'")

    if plot:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(time, velocity, label="Velocity", color="navy")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Velocity Time History")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(time, displacement, label="Displacement", color="darkorange")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.title("Displacement Time History")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    return time, velocity, displacement


def plot_time_series_from_column(file_path: str, dt: float, ylabel: str = "Value", title: str = "Time Series", scale: float = 1.0):
    """
    Plots a time series from a single-column text file using a specified time step.

    Parameters:
        file_path (str): Path to the column-formatted file (1 value per line).
        dt (float): Time step between each sample.
        ylabel (str): Label for the Y-axis.
        title (str): Title of the plot.
    """
    # Load values
    data = np.loadtxt(file_path)*scale
    time = np.arange(0, len(data) * dt, dt)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(time, data, label="Time Series")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    


def compare_single_column_series(file1, file2, dt, label1="Series 1", label2="Series 2", ylabel="Value", title="Comparison of Time Series"):
    """
    Compares two single-column time series by plotting them together.

    Parameters:
        file1 (str): Path to the first file (1 column).
        file2 (str): Path to the second file (1 column).
        dt (float): Time step between values.
        label1 (str): Label for the first series.
        label2 (str): Label for the second series.
        ylabel (str): Y-axis label.
        title (str): Plot title.
    """
    y1 = np.loadtxt(file1)
    y2 = np.loadtxt(file2)

    t1 = np.arange(len(y1)) * dt
    t2 = np.arange(len(y2)) * dt

    plt.figure(figsize=(10, 5))
    plt.plot(t1, y1, label=label1, color="navy")
    plt.plot(t2, y2, label=label2, color="darkgreen", linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def compare_series(series1, series2, dt=0.02, label1='Static', label2='Transient'):
    time = np.arange(len(series1)) * dt
    min_len = min(len(series1), len(series2))
    
    plt.figure(figsize=(8, 5))
    plt.plot(time[:min_len], series1[:min_len], label=label1, linewidth=2)
    plt.plot(time[:min_len], series2[:min_len], label=label2, linewidth=2, linestyle='--')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]') # Changed from [in] to [m]
    plt.title('Comparison of displacement Time Histories')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def morsa_to_opensees(input_file, output_file, scale_factor=9.81, dt=None, plot=False): # Changed default scale_factor to 9.81 m/s^2 per g
    """
    Converts acceleration time history from g to m/sec² and saves it for OpenSees.

    Parameters:
        input_file (str): Path to input file containing two columns: [time, acceleration in g].
        output_file (str): Path to output file to save converted acceleration (in m/sec²).
        scale_factor (float): Conversion factor (default is 9.81 m/sec² per g).
        dt (float or None): Time step between points. If None, extracted from file.
        plot (bool): Whether to plot the converted data with symmetric y-axis.

    Returns:
        dt (float): Time step between samples [s].
        npts (int): Number of data points.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the file with 2 columns: [time, acceleration]
    data = np.loadtxt(input_file)
    time = data[:, 0]
    accel_g = data[:, 1]
    npts = len(accel_g)

    # Infer time step if not given
    if dt is None:
        if len(time) >= 2:
            dt = time[1] - time[0]
        else:
            raise ValueError("Time step could not be inferred from data — provide 'dt' manually.")

    # Convert from g to m/sec²
    accel_converted = accel_g * scale_factor

    # Save only the acceleration (no time)
    np.savetxt(output_file, accel_converted, fmt='%.6e')
    print(f"✅ Converted file saved to '{output_file}'")

    if plot:
        max_val = np.max(np.abs(accel_converted))
        time_axis = np.arange(npts) * dt
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, accel_converted, label='Acceleration [m/sec²]')
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration [m/sec²]")
        plt.title("Converted Ground Motion")
        plt.ylim([-max_val, max_val])  # Symmetric y-axis
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return dt, npts



def compute_response_spectrum(accel, dt, periods, damping_ratio=0.05):
    """
    Computes the pseudo-acceleration response spectrum for a given acceleration time series.

    Parameters:
        accel (array-like): Ground acceleration in m/sec².
        dt (float): Time step in seconds.
        periods (array-like): List or array of periods (sec) to compute the spectrum.
        damping_ratio (float): Damping ratio (e.g., 0.05 for 5%).

    Returns:
        Sa (np.ndarray): Pseudo-acceleration spectrum [m/sec²].
    """
    import numpy as np

    omega_n = 2 * np.pi / np.clip(periods, 0.02, None)  # Avoid zero or near-zero periods
    Sa = np.zeros_like(periods)

    for i, wn in enumerate(omega_n):
        m = 1.0
        k = m * wn**2
        c = 2 * damping_ratio * m * wn

        u = 0.0
        v = 0.0
        a_s = 0.0
        u_hist = []

        for ag in accel:
            p = -m * ag
            a = (p - c*v - k*u) / m
            v += a * dt
            u += v * dt
            a_s = wn**2 * u # Pseudo-acceleration
            u_hist.append(abs(a_s))

        Sa[i] = max(u_hist)

    return Sa

def plot_response_spectrum(accel_file, dt, output_file=None, damping_ratio=0.05):
    """
    Reads acceleration from a file, computes, and plots the pseudo-acceleration response spectrum.

    Parameters:
        accel_file (str): Path to acceleration file (1 column, in m/sec²).
        dt (float): Time step in seconds.
        output_file (str): If provided, saves the figure to file.
        damping_ratio (float): Damping ratio for the spectrum.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    accel = np.loadtxt(accel_file)
    periods = np.linspace(0.02, 5.0, 100)  # Linear period scale

    Sa = compute_response_spectrum(accel, dt, periods, damping_ratio)

    plt.figure(figsize=(8, 5))
    plt.plot(periods, Sa, linewidth=2)
    plt.xlabel("Period [s]")
    plt.ylabel("Pseudo-acceleration [m/sec²]")
    plt.title(f"P-SA Response Spectrum (ξ = {damping_ratio*100:.1f}%)")
    plt.grid(True)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"✅ Response spectrum saved to '{output_file}'")
    else:
        plt.show()