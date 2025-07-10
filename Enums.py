'''
This module contains Enums, constants and other static objects that are used in other modules
'''

from enum import Enum

DEBUG = False # Whether to print debug info into console

class Data_Process_Options(Enum):
    Raw = 0,
    Downsample = 1,
    Average = 2,
    Smoothing = 3,
    Differentiation = 4,
    Normalization = 5,
    FFT = 6,
    DP = 7

'''
    :param Label: Label
    :param Bal: Balance - baseline signal measurement
    Stokes parameters
    :param S0: total intensity of the light
    :param S1: linear polarization along horizontal and vertical axes
    :param S2: linear polarization along axes at 45° to the horizontal
    :param S3: circular polarization component of the light
'''
RELEVANT_KEYS = [ 'Label', 'Bal', 'S0', 'S1', 'S2', 'S3' ]
NON_LABEL_KEYS = [ 'Bal', 'S0', 'S1', 'S2', 'S3' ]
STOKES_PARAMETERS = [ 'S0', 'S1', 'S2', 'S3' ]
SEC_PARAMETERS = ['DOP', 'DOLP', 'DOCP', 'AOLP']