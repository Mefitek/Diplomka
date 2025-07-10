'''
This module contains functions that are used in other modules
'''

from typing import Dict, List
from Enums import *
import os
import numpy as np # for Data Processing
import h5py # for working with HDF5 files
from scipy.signal import butter, filtfilt # for lowpass filter
import matplotlib.pyplot as plt # 2D plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
from matplotlib.widgets import Slider # 3D plotting
import easygui # choosing a file on disk

#========================================
#====        DATA PROCESSING         ====
#========================================

def process_dataset(dataset: Dict[str, np.ndarray],
                     relevant_keys: List[str],
                     process_option: Data_Process_Options,
                     *args) -> Dict[str, np.ndarray] :
    '''
    Applies simple processing function on the given dataset
    :param dataset: dataset to be processed
    :param relevant_keys: relevant keys for given dataset
    :param process_option: how to process the dataset
    :param *args: additional arguments for processing needs
    :return: processed_dataset (dict of numpy ndarrays)
    '''
    if not args or args[0] is None:
        if process_option not in (Data_Process_Options.Differentiation, Data_Process_Options.Normalization):
            raise ValueError("Please enter a numerical argument for given Processing option")
    
    processed_dataset = dataset
    for key in relevant_keys:
        data = dataset[key]
    # Downsample
        if process_option == Data_Process_Options.Downsample:
            step = args[0]
            data = data[::step]    
    # Average
        elif process_option == Data_Process_Options.Average:
            chunk_size = args[0] 
            data = np.mean(data[:len(data) // chunk_size * chunk_size].reshape(-1, chunk_size), axis=1)      
    # Smoothing
        elif process_option == Data_Process_Options.Smoothing:
            window_size = args[0]
            if window_size < 2:
                raise ValueError("Window size must be at least 2 for smoothing.")
            data = np.convolve(data, np.ones(window_size) / window_size, mode='valid') 
    # Differentiation
        elif process_option == Data_Process_Options.Differentiation:
            data = np.diff(data)
    # Normalization
        elif process_option == Data_Process_Options.Normalization:
            if not args or args[0] is None: # default normalization
                min_val, max_val = np.min(data), np.max(data) 
            else: # normalize by global min/max values
                min_val = args[0][key]['min']
                max_val = args[0][key]['max']
                
            if max_val - min_val == 0:
                data = np.zeros_like(data)
            else:
                data = (data - min_val) / (max_val - min_val)
    # FFT
        elif process_option == Data_Process_Options.FFT:
            window_size, shift = args[:2]
            if window_size < 2 or shift < 1:
                raise ValueError("Window size must be at least 2 and shift must be at least 1.")
            fft_result = []
            for i in range(0, len(data) - window_size + 1, shift):
                window = data[i : i+window_size]
                window = np.abs(np.fft.rfft(window))
                window = 10 * np.log10(window + 1e-12) # [dB]
                fft_result.append(window)
            data = np.array(fft_result)
    # Low-Pass Filter (DP)
        elif process_option == Data_Process_Options.DP:
            cutoff_freq = args[0]
            sampling_freq = 1.0
            if len(args) > 1:
                sampling_freq = args[1]
            data = lowpass_filter(data, cutoff_freq, sampling_freq)
        processed_dataset[key] = data
    
    return processed_dataset

def lowpass_filter(data, cutoff, fs, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                return filtfilt(b, a, data)

def calc_secondary_parameters(dataset: Dict[str, np.ndarray], relevant_keys: list, params: list):
    """
    TODO: Update doc comment to include other parameters
    Compute Degree of Polarization (DOP) and add it to the dataset.
    
    :param dataset: Dictionary containing loaded HDF5 data.
    :param relevant_keys: List of relevant keys from the dataset.
    :return: Updated dataset with the 'DOP' key added.
    """
    if all(key in dataset for key in ['S0', 'S1', 'S2', 'S3']):
        S0 = dataset['S0']
        S1 = dataset['S1']
        S2 = dataset['S2']
        S3 = dataset['S3']
        
        nonzero_mask = S0 != 0

        # DOP
        if 'DOP' in params:
            DOP = np.zeros_like(S0, dtype=np.float64)
            DOP[nonzero_mask] = np.sqrt(S1[nonzero_mask]**2 + S2[nonzero_mask]**2 + S3[nonzero_mask]**2) / S0[nonzero_mask]
            dataset['DOP'] = DOP
            relevant_keys.append('DOP')
        # DOLP
        if 'DOLP' in params:
            DOLP = np.zeros_like(S0, dtype=np.float64)
            DOLP[nonzero_mask] = np.sqrt(S1[nonzero_mask]**2 + S2[nonzero_mask]**2) / S0[nonzero_mask]
            dataset['DOLP'] = DOLP
            relevant_keys.append('DOLP')
        # DOCP
        if 'DOCP' in params:
            DOCP = np.zeros_like(S0, dtype=np.float64)
            DOCP[nonzero_mask] = S3[nonzero_mask] / S0[nonzero_mask]
            dataset['DOCP'] = DOCP
            relevant_keys.append('DOCP')
        # AOLP
        if 'AOLP' in params:
            nonzero_mask = S1 != 0
            AOLP = np.zeros_like(S1, dtype=np.float64)
            AOLP[nonzero_mask] = 0.5*np.arctan(S2[nonzero_mask] / S1[nonzero_mask])
            dataset['AOLP'] = AOLP
            relevant_keys.append('AOLP')

    else:
        print("Warning: Missing required keys (S0, S1, S2, S3) for DOP calculation.")
    
    return dataset, relevant_keys

#========================================
#====      DATA VISUALISATION        ====
#========================================

def get_hdf_keys(hdf_file: h5py._hl.files.File) -> h5py._hl.base.KeysViewHDF5:
    '''
    Load keys (names of group members) from HDF5 file
    :param hdf_file: opened HDF5 file
    :return: keys from the HDF5 file
    '''
    keys = hdf_file.keys()
    if DEBUG: print(list(keys))
    return keys

def dataset_vis_2D(dataset: Dict[str, np.ndarray],
            relevant_keys: List[str],
            show_metadata: bool,
            metadata: Dict[str, str],
            save_as_pdf: bool = False) -> None:
    '''
    Visualises the HDF5 data using PyPlot (Basic 2D visualisation)
    :param dataset: dataset to be visualised
    :param relevant_keys: List of keys to plot from the dataset
    :param metadata: dictionary with metadata information
    :param save_as_pdf: whether to save figure as a pdf
    :return: nothing
    '''
    if show_metadata:
        fig, axs = plt.subplots(figsize=(12, 6), ncols=2, gridspec_kw={'width_ratios': [5, 1.5]})
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        axs = [ax]

    axs[0].set_title("Overlayed Data of HDF5 file", fontsize=14)
    for key in relevant_keys:
        data = dataset[key]
        if data.ndim == 1:
            axs[0].plot(data, label=key)
        elif data.ndim == 2:
            axs[0].plot(data.mean(axis=0), label=key)
        else:
            print(f"Data for {key} has more than 2 dimensions; unable to plot directly.")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("Value")
    axs[0].legend()

    if show_metadata:
        axs[1].axis('off')
        table_data = [[key, value] for key, value in metadata.items()]
        table = axs[1].table(cellText=table_data, loc="center", cellLoc="left", colLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width([0, 1])
        for cell in table.get_celld().values():
            cell.set_height(0.075)

    plt.tight_layout()
    plt.show()

    if save_as_pdf:
        output_path = easygui.filesavebox(
            msg="Choose file path",
            default="plot.pdf",
            filetypes=["*.pdf"]
        )
        if output_path:
            fig.savefig(output_path, format='pdf')
            print("Plot saved as PDF:", output_path)
        else:
            print("Save operation cancelled.")
    return

def dataset_vis_3D(dataset: Dict[str, np.ndarray],
                   save_as_pdf: bool = False
                   ) -> None:
    '''
    Visualises the HDF5 data using matplotlib (3D visualisation with Poincaré sphere)
    :param dataset: dataset to be visualised
    :param save_as_pdf: whether to save figure as a pdf
    :return: nothing
    '''
    if not all(key in dataset for key in STOKES_PARAMETERS):
        raise ValueError("Dataset must contain the keys 'S0', 'S1', 'S2', and 'S3' for visualization on the Poincaré sphere.")

    S0 = dataset['S0']
    S1 = dataset['S1']
    S2 = dataset['S2']
    S3 = dataset['S3']

    if not (S0.shape == S1.shape == S2.shape == S3.shape):
        raise ValueError("Stokes parameters S0, S1, S2, and S3 must have be of the same length.")

    s1_norm = S1 / S0
    s2_norm = S2 / S0
    s3_norm = S3 / S0

    fig = plt.figure(figsize=(10, 8)) # Creating a 3D plot
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the surface of the Poincaré sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Help function to plot the point and vector inside the sphere
    def draw_poin_sphere(ax: Axes3D,
                     x: np.ndarray,
                     y: np.ndarray,
                     z: np.ndarray,
                     s1_norm: np.float32,
                     s2_norm: np.float32,
                     s3_norm: np.float32
                     ) -> None:
        ax.set_title("Poincaré sphere - Stokes parameters (interactive)")
        ax.set
        ax.set_xlabel('S1')
        ax.set_ylabel('S2')
        ax.set_zlabel('S3')
        

        ax.plot_surface(x, y, z, color='lightgrey', alpha=0.3, edgecolor='none')
        
        point, = ax.plot([s1_norm], [s2_norm], [s3_norm],
                        'ro', markersize=10, label="Stokes vector (data point)") 
        arrow = ax.quiver(0, 0, 0, s1_norm, s2_norm, s3_norm,
                        color='red', linewidth=2, arrow_length_ratio=0.2, label="Stokes vector (arrow)")
        
        ax.legend()

    draw_poin_sphere(ax, x, y, z, s1_norm[0], s2_norm[0], s3_norm[0])

    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03], facecolor='lightgrey')
    slider = Slider(ax_slider, 'Sample index', 0, len(S1) - 1, valinit=0, valstep=1)

    # Callback function for updating the point and arrow (has to be cleared to be redrawn)
    def update(val):
        index = int(slider.val)
        ax.clear()
        draw_poin_sphere(ax, x, y, z, s1_norm[index], s2_norm[index], s3_norm[index])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    # save as pdf
    if save_as_pdf:
        output_path = easygui.filesavebox(
            msg="Choose file path",
            default="plot.pdf",
            filetypes=["*.pdf"]
        )
        if output_path:
            fig.savefig(output_path, format='pdf')
            print("Plot saved as PDF:", output_path)
        else:
            print("Save operation cancelled.")
    return

def dataset_vis_spectogram(fft_data, sampling_rate, window_size, shift):
    '''
    Plot spectrogram from precomputed FFT data
    :param fft_data: 2D NumPy array (n_windows x n_frequencies)
    :param sampling_rate: Sampling rate of the signal in Hz
    :param window_size: Window size used for FFT
    :param shift: Step size between consecutive windows
    '''
    n_windows = fft_data.shape[0]
    freqs = np.fft.rfftfreq(window_size, d=1/sampling_rate)
    times = np.arange(n_windows) * (shift / sampling_rate)

    plt.figure(figsize=(10, 6))
    #plt.pcolormesh(times, freqs, fft_data.T, shading='gouraud', vmin=-10, vmax=30) # TODO: Kvantil
    plt.pcolormesh(times, freqs, fft_data.T, shading='auto')
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.show()

#========================================
#=====            OTHER             =====
#========================================

def load_hdf_data(file_path: str):
    '''
    Load data from HDF5 file
    :param file_path: path to the HDF5 file
    :return: dataset (dict of numpy ndarrays), keys (list of strings)
    '''
    with h5py.File(file_path) as hdf_file:
        if DEBUG:
            print(f"Opened file: {file_path}")
        keys = get_hdf_keys(hdf_file)
        relevant_keys = [key for key in keys if key in RELEVANT_KEYS]
        dataset = {}
        for key in relevant_keys:
            data = hdf_file[key][()]
            dataset[key] = data
        return dataset, relevant_keys

def get_description_from_filename(file_path: str):
    '''
    Getsf file description if it followes given naming convention
    :param file_path: path to the HDF5 file
    :return: desciption (dict of strings) and sampling frequency (integer)
    '''
    filename = os.path.basename(file_path)
    descr = {}
    parts = filename.split("_")

    # Dictionaries 
    fiber_rename = {
        "g652": "G.652",
        "g657": "G.657" }

    fiber_stress = {
        "klep": "Knocking",
        "N1": "Force - Pointed end effector",
        "N2": "Force - Rounded end effector",
        "N3": "Movement - Hook end effector" }
    
     # Measured optic fiber
    fiber = fiber_rename.get(parts[0], parts[0])
    descr.update({"Fiber":fiber})

    # Measured data line
    line = parts[1]
    descr.update({"Data line":line})

    # Fiber stressed by
    stress = fiber_stress.get(parts[2], parts[2])
    descr.update({"Stress":stress})
        
        # Stress - Knocking
    knock_sides = {
        "L": "Left (with coating)",
        "R": "Right (no coating)" }

    knock_item = {
        "prav": "Ruler (narrow side)",
        "prav2": "Ruler (wide side)",
        "prst": "Finger",
        "sroub": "Screwdriver",
        "mobil": "Phone vibrations",
        "suplik": "Drawer" }

    if stress == fiber_stress["klep"]:
        if parts[3] == "klid":
            descr.update({"State":"Idle"})
        elif parts[3] == "R" or parts[3] == "L":
            side = knock_sides.get(parts[3], parts[3])
            descr.update({"Side":side})
            item = knock_item.get(parts[4], parts[4])
            descr.update({"Item":item})
        else:
            item = knock_item.get(parts[3], parts[3])
            descr.update({"Item":item})
        
        # Stress - Movement
    elif stress == fiber_stress["N3"]:
        if parts[4] == "klid":
            descr.update({"State":"Idle"})
        else:
            reps = parts[3].replace('r', '')
            descr.update({"Repetitions":reps})
            delay = parts[4].replace('t', '') + ' s'
            descr.update({"Delay":delay})
        
        # Stress - Force
    else:
        if parts[5] == "klid":
            descr.update({"State":"Idle"})
        else:
            reps = parts[3].replace('r', '')
            descr.update({"Repetitions":reps})
            time = parts[4].replace('t', '') + ' s'
            descr.update({"Applied for":time})
            force = parts[5].replace('F', '') + ' N'
            descr.update({"Force":force})

    descr.update({"Date":parts[-5]})
    # Timestamp is in parts[-5] to parts[-2]
    fs = (parts[-1].split("."))[0]
    fs = int(fs)
    return descr, fs

def choose_file():
    file_path = easygui.fileopenbox(
        title="Choose a .hdf5 file",
        default="*.hdf5",
        filetypes=["*.hdf5"]
    )
    return file_path


#========================================
#=====            TEST              =====
#========================================


