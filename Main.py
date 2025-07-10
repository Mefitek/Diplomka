'''
This module serves as the entry point for the application. 
Its primary purpose is to coordinate the program's main functionality and test it.
'''

from Classes import *
import os
import sys

#D = Data(file_path)
#D.visualize_2D()
#D.process(Data_Process_Options.DP, 0.1*D.sampling_freq, D.sampling_freq)
#D.process(Data_Process_Options.Downsample, D.fs)
#D.process(Data_Process_Options.Downsample, 10)
#D.process(Data_Process_Options.FFT, 4096, 1024)
#print(D.description)
#D.calc_secondary_parameters(SEC_PARAMETERS)
#D.calc_secondary_parameters(['AOLP'])
#D.calc_secondary_parameters(['DOP', 'DOLP', 'DOCP'])

#D.process(Data_Process_Options.Downsample, 100)

#D.visualize_2D()
#D.visualize_3D()

# FFT Test
#D.process(Data_Process_Options.FFT, 4096, 1024)
#D.process(Data_Process_Options.FFT, 1024, 256)
#D.visualize_spectogram('S0')

#path = choose_file()

#path = r'c:\\Users\\mefit\\Desktop\\DP\\Mereni\\ALL_IN_ONE_TRIMMED\\g652_200G_N1\\g652_200G_N1_r5_t15_F24_2024-10-01_11-11-59.170672_2024-10-01_11-15-01.431444_13086.hdf5'
#D = Data(path)

#D.process(None, Data_Process_Options.DP, 0.0015 * D.sampling_freq, D.sampling_freq)
#D.process(None, Data_Process_Options.Average, 100)
#D.visualize_2D(show_metadata=True, save_as_pdf=True)
#D.visualize_3D(save_as_pdf=True)
