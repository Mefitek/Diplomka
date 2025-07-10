'''
This module contains scripts that assist with managing files and data sets
'''
import os
import h5py
from Classes import *
import torch
import numpy as np
import random
from collections import defaultdict
from Functions import *


# Set the directory containing your files

def replace():
    folder_path = r'C:\Users\mefit\Desktop\DP\Mereni\2024_10_01_Data_mereni\g652_200G_N1'

    replace_what = '100G'
    replace_by = '200G'

    # Iterate through all files in the specified directory
    for filename in os.listdir(folder_path):
        # Check if the filename contains 'n2' to ensure we only rename relevant files
        if replace_what in filename:
            # Create the new filename by replacing 'n2' with 'N2'
            new_filename = filename.replace(replace_what, replace_by)
            
            # Construct the full file paths
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {filename} -> {new_filename}')

def add_ns():
    folder_path = r'C:\Users\mefit\Desktop\DP\Mereni\2024_10_01_Data_mereni\g652_100G_N1'

    for filename in os.listdir(folder_path):
        if 'G' in filename:
            index = filename.index('G')
            new_filename = filename[:index + 1] + '_N1' + filename[index + 1:] # new filename = _N1 after the first G
            
            # Construct new file path
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {filename} -> {new_filename}')

def show_last_N():
    folder_path = r'C:\Users\mefit\Desktop\DP\Mereni\ALL_IN_ONE'
    folder_path_new = r'C:\Users\mefit\Desktop\DP\Mereni\ALL_IN_ONE_TRIMMED'
    folder_path = folder_path_new
    test_file = 'g652_200G_N1_klid_2024-09-30_10-45-12.125146_2024-09-30_10-48-20.467618_13082.hdf5'
    file_path = os.path.join(folder_path, test_file)
    N = 60

    # Otevření souboru a čtení posledních 500 hodnot
    with h5py.File(file_path, 'r') as f:
        if 'Bal' in f:
            data = f['Bal']
            last_N_values = data[-N:]  # Posledních 500 hodnot
            print(last_N_values)
        else:
            print("Key 'Bal' doesn't exist in file.")

def trim_zeros():
    folder_path = r'C:\Users\mefit\Desktop\DP\Mereni\ALL_IN_ONE'
    folder_path_new = r'C:\Users\mefit\Desktop\DP\Mereni\ALL_IN_ONE_TRIMMED'
    N = 60

    # Projít všechny soubory ve složce a podsložkách
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.hdf5'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                new_file_path = os.path.join(folder_path_new, relative_path)
                new_file_dir = os.path.dirname(new_file_path)
                
                # Vytvoření cílové složky, pokud neexistuje
                os.makedirs(new_file_dir, exist_ok=True)

                # Otevření původního souboru pro čtení a nového pro zápis
                with h5py.File(file_path, 'r') as f_src, h5py.File(new_file_path, 'w') as f_dst:
                    for key in f_src.keys():
                        data = f_src[key]
                        if isinstance(data, h5py.Dataset):  # Ověření, že jde o dataset
                            trimmed_data = data[:-N]  # Ořezání posledních N hodnot
                            f_dst.create_dataset(key, data=trimmed_data)  # Uložení oříznutého datasetu
                        else:
                            f_src.copy(key, f_dst)  # Kopírování ostatních objektů

                print(f"Nový soubor byl uložen jako {new_file_path}")

def label_a_file(do_label=False):
    
    def label_data(dataset: Dict[str, np.ndarray]) -> None:
        length = len(dataset['Label'])
        dataset['Label'][:] = 0  # Vynuluj všechny hodnoty

        while True:
            try:
                start = int(input("Zadej začátek intervalu (nebo -1 pro ukončení): "))
                if start == -1:
                    break
                end = int(input("Zadej konec intervalu: "))

                start = start*100
                end = end*100

                if 0 <= start < end <= length:
                    dataset['Label'][start:end] = 1
                else:
                    print(f"Neplatný interval. Musí být v rozsahu 0 až {length} a start < end.")
            except ValueError:
                print("Zadej platné celé číslo.")

        print("Data souboru byla označena.")
    def make_new_path(path: str) -> str:
        return path.replace("ALL_IN_ONE_TRIMMED", "ALL_IN_ONE_LABELED")

    print("Label a HDF5 file")
    file_path = choose_file()
    D = Data(file_path)

    if not do_label: #only read
        # for lowpass the parameter is best at 0.015 according to my observations
        # 0.0015 worked better for 200G data tho
        #D.process(NON_LABEL_KEYS, Data_Process_Options.DP, 0.0015 * D.sampling_freq, D.sampling_freq)
        #D.process(NON_LABEL_KEYS, Data_Process_Options.Average, 100)
        #D.process(['Label'], Data_Process_Options.Downsample, 100)
        D.visualize_2D(show_metadata=False)

    else:
        label_data(D.dataset)
        new_path = make_new_path(D.path)
        D.save_dataset_as(new_path)
        D.process(Data_Process_Options.Downsample, 10)
        D.visualize_2D()

def check_CUDA_avail():
    import torch
    import torchvision
    import sys
    print(f"\n\tpython version:\t{sys.version}")
    print(f"\ttorch version:\t{torch.__version__}")
    print(f"\ttorchvision v:\t{torchvision.__version__}")
    print(f"\tCUDA version:\t{torch.version.cuda}")
    print(f"\tCUDA device:\t{torch.cuda.get_device_name(0)}")
    print(f"\tCUDA available:\t{torch.cuda.is_available()}\n")

def preprocess_data(use_labels=True, folder_path='C:\\Users\\mefit\\Desktop\\DP\\Mereni\\ALL_IN_ONE_LABELED'):
    folder_path_new = r'C:\\Users\\mefit\\Desktop\\DP\\Python\\dp\\pytorch\\data\\hdf5'

    # Vytvoření cílové složky pokud neexistuje
    os.makedirs(folder_path_new, exist_ok=True)

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.hdf5'):
                file_path = os.path.join(root, file)
                D = Data(file_path) # 🧠 Zpracování pomocí třídy Data
                # ========================================
                # ========  PREPROCESSING OPTIONS  =======
                if 'Bal' in D.dataset:
                    del D.dataset['Bal']
                #if 'S0' in D.dataset:
                #    del D.dataset['S0']
                #    del D.dataset['S1']
                #    del D.dataset['S2']
                #    del D.dataset['S3']
                if (not use_labels) and 'Label' in D.dataset:
                    del D.dataset['Label']
                DECIMATION_RATE = 10

                #D.process(['Bal'], Data_Process_Options.DP, 0.0015 * D.sampling_freq, D.sampling_freq)
                #D.process(['Bal'], Data_Process_Options.Average, DECIMATION_RATE)
                D.process(['S0', 'S1', 'S2', 'S3'], Data_Process_Options.DP, 0.0015 * D.sampling_freq, D.sampling_freq)
                D.process(['S0', 'S1', 'S2', 'S3'], Data_Process_Options.Average, DECIMATION_RATE)
                #D.process(['S0', 'S1', 'S2', 'S3'], Data_Process_Options.Normalization, normalization_arg)
                if use_labels:
                    D.process(['Label'], Data_Process_Options.Downsample, DECIMATION_RATE)
                # ========================================

                new_file_path = os.path.join(folder_path_new, os.path.basename(file_path))
                D.save_dataset_as(new_file_path)
                #print(f"Preprocessed file saved as: {new_file_path}")
    print("\n\nAll HDF5 files preprocessed!\n")

def convert_hdf5_to_pt(
    input_dir,
    output_file,
    window_size=2048,
    stride=512,
    threshold=0.1,
    channels=('Bal', 'S0', 'S1', 'S2', 'S3'),
    labels_exist=True,
    seed=42
):
    X_data = []
    Y_data = []
    label_key = 'Label'

    hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    random.seed(seed)
    random.shuffle(hdf5_files)  # <- shuffle file order

    print(f"Found {len(hdf5_files)} HDF5 files in '{input_dir}' (shuffled).")

    for idx, filename in enumerate(hdf5_files):
        print(f"[{idx+1}/{len(hdf5_files)}] Processing {filename}...")
        path = os.path.join(input_dir, filename)

        with h5py.File(path, 'r') as f:
            if labels_exist:
                lengths = [len(f[ch]) for ch in channels + (label_key,)]
            else:
                lengths = [len(f[ch]) for ch in channels]
            min_len = min(lengths)

            for start in range(0, min_len - window_size + 1, stride):
                end = start + window_size
                signals = [f[ch][start:end] for ch in channels]
                x = np.stack(signals, axis=0)  # [C, window_size]

                if labels_exist:
                    y_window = f[label_key][start:end]
                    ratio = np.mean(y_window >= 1.0)
                    y_max = int(np.max(y_window))
                    y = int(ratio >= threshold)*y_max
                    Y_data.append(torch.tensor(y, dtype=torch.long))
                X_data.append(torch.tensor(x, dtype=torch.float32))

    # Save result
    if labels_exist:
        dataset = {'inputs': torch.stack(X_data), 'labels': torch.tensor(Y_data)}
    else:
        dataset = {'inputs': torch.stack(X_data)}
    torch.save(dataset, output_file)
    print(f"\nSaved dataset with {len(X_data)} samples to: {output_file}")

def split_pt_dataset(pt_path, output_dir, labels_exist = True, val_ratio=0.1, test_ratio=0.1, seed=69):
    os.makedirs(output_dir, exist_ok=True)
    data = torch.load(pt_path)
    inputs = data['inputs']
    if labels_exist:
        labels = data['labels']
        assert len(inputs) == len(labels), "Mismatched inputs and labels length"

    total = len(inputs)
    indices = list(range(total))
    random.seed(seed)
    random.shuffle(indices)

    test_len = int(total * test_ratio)
    val_len = int(total * val_ratio)
    train_len = total - val_len - test_len

    train_idx = indices[:train_len]
    val_idx = indices[train_len:train_len + val_len]
    test_idx = indices[train_len + val_len:]

    def subset(idx_list, labels_exist):
        if labels_exist:
            return {'inputs': inputs[idx_list], 'labels': labels[idx_list]}
        return {'inputs': inputs[idx_list]}

    torch.save(subset(train_idx, labels_exist=labels_exist), os.path.join(output_dir, 'dataset_train.pt'))
    torch.save(subset(val_idx, labels_exist=labels_exist), os.path.join(output_dir, 'dataset_val.pt'))
    torch.save(subset(test_idx, labels_exist=labels_exist), os.path.join(output_dir, 'dataset_test.pt'))

    print(f"\n\nSplit completed:")
    print(f"\tTrain: {train_len} samples")
    print(f"\tVal:   {val_len} samples")
    print(f"\tTest:  {test_len} samples")

def compute_pos_weight(pt_relative_path):
    data = torch.load(pt_relative_path)
    labels = data['labels']  # shape: [N]

    positives = (labels == 1).sum().item()
    negatives = (labels == 0).sum().item()

    if positives == 0:
        raise ValueError("No positive samples in dataset, pos_weight is undefined.")

    pos_weight = negatives / positives

    print(f"Positive samples: {positives}")
    print(f"Negative samples: {negatives}")
    print(f"Computed pos_weight: {pos_weight:.4f} (1/pos_weight = {(1/pos_weight):.4f})")

def find_minmax_in_hdf5(folder_path):
    min_vals = defaultdict(lambda: np.inf)
    max_vals = defaultdict(lambda: -np.inf)

    for root, _, files in os.walk(folder_path):
        for filename in sorted(files):
            if not filename.endswith('.hdf5'):
                continue

            path = os.path.join(root, filename)
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    data = f[key][:]
                    min_vals[key] = min(min_vals[key], np.min(data))
                    max_vals[key] = max(max_vals[key], np.max(data))

    # Return as dictionary
    key_dict = defaultdict()
    for key in min_vals:
        key_dict[key] = {'min': min_vals[key], 'max': max_vals[key]}
    return key_dict
    
def filter_out_label_1(input_pt_path, output_pt_path):
    if not os.path.exists(input_pt_path):
        raise FileNotFoundError(f"Input file not found: {input_pt_path}")

    data = torch.load(input_pt_path)
    inputs = data['inputs']  # shape: [N, C, T]
    labels = data['labels']  # shape: [N]

    # Keep only samples where label = 0
    mask = labels == 0.0
    filtered_inputs = inputs[mask]
    filtered_labels = labels[mask]

    print(f"Original samples: {len(labels)}")
    print(f"Remaining after filter: {len(filtered_labels)}")

    torch.save({
        'inputs': filtered_inputs,
        'labels': filtered_labels
    }, output_pt_path)

    print(f"Filtered dataset saved to: {output_pt_path}")

def remove_labels_from_pt(input_pt_path, output_pt_path):
    if not os.path.exists(input_pt_path):
        raise FileNotFoundError(f"Input file not found: {input_pt_path}")

    data = torch.load(input_pt_path)

    # Create new dataset with only inputs
    new_data = {
        'inputs': data['inputs']
    }

    torch.save(new_data, output_pt_path)
    print(f"New dataset without labels saved to: {output_pt_path}")

def adjust_labels(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.endswith('.hdf5'):
                continue

            full_input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_input_path, input_dir)
            full_output_path = os.path.join(output_dir, relative_path)

            os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

            with h5py.File(full_input_path, 'r') as f_in, h5py.File(full_output_path, 'w') as f_out:
                # Copy dataset except for Label
                for key in f_in.keys():
                    if key != 'Label':
                        f_out.create_dataset(key, data=f_in[key][:])

                labels = f_in['Label'][:]
                modified_labels = labels.copy()

                fname = os.path.splitext(filename.lower())[0]

                # Label adjustement rules
                # 0 = Calm state
                if ('klid' in fname) or ('f0' in fname):
                    pass
                # 1 = Applied force 2-10 Newton
                elif any(f"f{n}" in fname for n in [2, 4, 6, 8, 10]):
                    pass
                # 3 = Applied force 12-40 Newton
                elif 'f' in fname:
                    modified_labels[labels == 1] = 3
                # 4 = Fiber movement
                elif 'n3' in fname:
                    modified_labels[labels == 1] = 4
                # 2 = Drawer shutting
                elif ('klep' in fname) and ('suplik' in fname):
                    modified_labels[labels == 1] = 2
                # 5 = Other fiber shocks
                elif 'klep' in fname:
                    modified_labels[labels == 1] = 5
                else:
                    print(f"\n\n\tUnrecognized filename pattern: {filename}\n\n")

                f_out.create_dataset('Label', data=modified_labels)


USE_LABEL = True
#preprocess_data(use_labels=USE_LABEL, folder_path='C:/Users/mefit/Desktop/DP/Mereni/ALL_IN_ONE_LABELED')
convert_hdf5_to_pt(input_dir='pytorch/data/hdf5', output_file='pytorch/data/pt/dataset.pt', channels = ('S0', 'S1', 'S2', 'S3'), threshold=0.0001, labels_exist = USE_LABEL, seed=1234)
#convert_hdf5_to_pt(input_dir='pytorch/data/hdf5', output_file='pytorch/data/pt/dataset.pt', channels = ('Bal',), threshold=0.0001, labels_exist = USE_LABEL, seed=1234)
#filter_out_label_1(input_pt_path = 'pytorch/data/pt/dataset.pt', output_pt_path = 'pytorch/data/pt/dataset_13_del.pt')
#remove_labels_from_pt(input_pt_path = 'pytorch/data/pt/dataset_13_del.pt', output_pt_path = 'pytorch/data/pt/dataset_13.pt')
#split_pt_dataset(pt_path='pytorch/data/pt/dataset.pt', output_dir='pytorch/data/pt', labels_exist = USE_LABEL, val_ratio=0.0, test_ratio=1.0)

#check_CUDA_avail()
#label_a_file(do_label = False) # True = label; False = read
