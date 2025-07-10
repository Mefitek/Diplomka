'''
This module contains Classes and other more complex data type variables that are used in other modules
'''

from Functions import *
from Enums import *

class Data:

    path : str
    dataset : Dict[str, np.ndarray]
    relevant_keys : List[str]
    process_option : Data_Process_Options
    description : Dict[str, str]
    sampling_freq : int
    fft_window_size : int
    fft_shift : int

    def __init__(self, hdf_filepath:str):
        self.path = hdf_filepath
        self.description, self.sampling_freq = get_description_from_filename(hdf_filepath)
        self.dataset, self.relevant_keys = load_hdf_data(hdf_filepath)
        self.process_option = Data_Process_Options.Raw.name

    def visualize_2D(self, show_metadata:bool=True, save_as_pdf:bool=False):
        dataset_vis_2D(self.dataset, self.relevant_keys, show_metadata, self.description, save_as_pdf)

    def visualize_3D(self, save_as_pdf:bool=False):
        dataset_vis_3D(self.dataset, save_as_pdf)
        
    def visualize_spectogram(self, key):
        dataset_vis_spectogram(self.dataset[key],
                             sampling_rate = self.sampling_freq,
                             window_size=self.fft_window_size,
                             shift=self.fft_shift)

    def process(self, keys, process_option: Data_Process_Options, *args):
        if keys == None: keys = self.relevant_keys
        self.dataset = process_dataset(self.dataset, keys, process_option, *args)
        self.process_option = process_option.name
        if process_option == Data_Process_Options.FFT:
            self.fft_window_size, self.fft_shift = args[:2]

    def calc_secondary_parameters(self, parameters):
        self.dataset, self.relevant_keys = calc_secondary_parameters(self.dataset, self.relevant_keys, parameters)

    def save_dataset_as(self, new_path: str) -> None:
        # Vytvoření cílové složky, pokud neexistuje
        new_file_dir = os.path.dirname(new_path)
        os.makedirs(new_file_dir, exist_ok=True)
        with h5py.File(new_path, 'w') as f:
            for key, value in self.dataset.items():
                f.create_dataset(key, data=value)
        print(f"\tDataset saved as: {new_path}")
        