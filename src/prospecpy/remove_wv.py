# Setting Up and Importing the Necessary Packages/Libraries
##Package for reading in Bruker OPUS type files
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from brukeropusreader import OpusData

##Package for subtraction algorithm
# Local Files
from .vaporfit import AtmFitParams, atm_subtraction


def subtract_wv(
    raw_spectra: OpusData, wv_spectrum: OpusData, show_plots=False
) -> tuple[list[AtmFitParams], np.ndarray]:
    """
    Performs water vapor subtraction on raw spectral data.

    Args:
        raw_spectra (OpusData): Raw spectral data (FTIR measurements).
        wv_spectrum (OpusData): Water vapor spectrum data.
        show_plots (bool, optional): Whether to display intermediate plots. Defaults to false.
    Returns:
        np.ndarray: Corrected spectral data after water vapor subtraction.
    Notes:
        - The function interpolates raw spectral data and water vapor spectrum to a common grid.
        - Water vapor spectrum is subtracted from raw spectral data to correct for water vapor absorption.
    """
    # Printing Contents and Size of Both Water Vapor and Raw Data
    # print(f'Parsed fields: '
    #   f'{list(raw_spectra.keys())}')
    # print(f'Size of Raw Spectra: 'f'{raw_spectra["AB"].size}')

    # print(f'Size of Water Vapor Spectrum: 'f'{wv_spectrum["AB"].size}')
    print(raw_spectra["Sample"]["SNM"][0:4])
    # Extracting the wavenumbers from the raw data
    raw_spectra_x = raw_spectra.get_range("AB")
    # the "AB" data can contain more null values at the end (at least 1)
    # so the getting useful data requires slicing the array:
    raw_spectra_abs = raw_spectra["AB"][0 : len(raw_spectra_x)]

    # Extracting the wavenumbers from the water vapor data
    wv_x = wv_spectrum.get_range("AB")
    # the "AB" data can contain more null values at the end (at least 1)
    # so the getting useful data requires slicing the array:
    wv_abs = wv_spectrum["AB"][0 : len(wv_x)]

    # Interpolating Data
    interpolated_raw_spectra = raw_spectra.interpolate(raw_spectra_x[0], raw_spectra_x[-1], 100)
    interpolated_wv_data = wv_spectrum.interpolate(wv_x[0], wv_x[-1], 100)

    # Plotting Individual elements
    if show_plots:
        print("Plotting interpolated AB")
        plt.plot(*raw_spectra.interpolate(raw_spectra_x[0], raw_spectra_x[-1], 100))
        plt.show()

        print("Plotting interpolated WV")
        plt.plot(*wv_spectrum.interpolate(wv_x[0], wv_x[-1], 100))
        plt.show()
    # Preparing the data for the subtraction algorithm
    ## Pulling Out Hydrogenase Wave Numbers and Absorbance
    raw_spectra_wavenb = interpolated_raw_spectra[0]
    raw_spectra_absorbance = interpolated_raw_spectra[1]
    ## Pulling Out Water Vapor Absorbance
    wv_absorbance = interpolated_wv_data[1]

    raw_sub_wv = atm_subtraction(raw_spectra_wavenb, raw_spectra_absorbance, wv_absorbance)

    return raw_sub_wv
