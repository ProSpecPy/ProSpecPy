import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

def baseline_spline(anchor_points, degree=3, smooth=0):
    """
    Function to fit a spline curve to anchor points data to estimate baseline.

    Parameters:
    - anchor_points: DataFrame
        DataFrame containing anchor points with 'wavenumber' and 'absorbance' columns.
    - degree: int, optional (default=3)
        Degree of the spline interpolation.
    - smooth: float, optional (default=0)
        Smoothing parameter for spline fitting.

    Returns:
    - baseline_curve: DataFrame
        DataFrame containing the fitted baseline curve with 'wavenumber' and 'absorbance' columns.
    """
    spline_fit = UnivariateSpline(anchor_points['wavenumber'], anchor_points['absorbance'],k = degree, s=smooth)
    x_range = np.linspace(int(min(anchor_points['wavenumber'])), int(max(anchor_points['wavenumber'])), 1000)
    #x_range = anchor_points['wavenumber']
    baseline_fit = spline_fit(x_range)
    baseline_curve = pd.DataFrame({'wavenumber':x_range, 'absorbance': baseline_fit})
    return baseline_curve



def baseline_correction(baseline_points, raw_wavenumber, raw_absorbance):
    """
    Perform baseline correction on raw absorbance data using baseline points.

    Args:
    - baseline_points (DataFrame): DataFrame containing baseline wavenumber and absorbance values.
    - raw_wavenumber (array): List of wavenumber values from raw data.
    - raw_absorbance (array): List of absorbance values from raw data.

    Returns:
    - baseline_corrected_abs (list): List of baseline-corrected absorbance values.
    """
    baseline_corrected_abs = []
    for idx, wv_num in enumerate(raw_wavenumber): #iterate through each data point in raw data
        #subtract the wv_num from the baseline[wavenumber] and take absolute of the diff and sort it and get the
        #idx of the point from the baseline that is closest to the raw datapoint
        diff_array = abs(baseline_points['wavenumber'] - wv_num)
        #this is the index of the closest wavenumber in the baseline curve to that of the datapoint
        closest_wv_num = diff_array.idxmin()
        #Now subtract the baseline absornace from the raw data absorbance
        raw_minus_baseline = raw_absorbance[idx] - baseline_points.loc[closest_wv_num, 'absorbance']

        if raw_minus_baseline <0:
            #if the difference is negative, then baseline point is higher than raw absorbance which is not possible. Hence appending 0 at those points
            baseline_corrected_abs.append(0)
        else:
            baseline_corrected_abs.append(raw_minus_baseline)

            
    return baseline_corrected_abs



def get_baseline_peak_index(baseline_corrected_abs, rawdata_wavenumber, raw_data_peak_wv):
    #get all the peaks index in the baselinecorrected data with no thresholds
    peak_index_baseline = find_peaks(baseline_corrected_abs)
    #get the corresponding wavenumbers present at the peak_index
    baseline_peak_wv = [[rawdata_wavenumber[i], i] for i in peak_index_baseline[0]]
    #Now obtain the corresponding peak wavenumbers using raw_data_peak_wv as reference. This was found using the 
    #raw spectra data
    range = 2
    peak_wv_baseline =[]
    peak_idx_baseline =[]
    for raw_wv in raw_data_peak_wv:
        ans = [ [wv[0], wv[1]] for wv in baseline_peak_wv if abs(wv[0] - raw_wv) <= range ]
        if ans:
            peak_wv_baseline.append(ans[0][0])
            peak_idx_baseline.append(ans[0][1])
    
    peak_baseline_abs = []
    for index in peak_idx_baseline:
        peak_baseline_abs.append(baseline_corrected_abs[index])
    
    return peak_idx_baseline, peak_wv_baseline, peak_baseline_abs

def plot_baseline_corrected_data(x_wavenb, baseline_abs, peak_wv, peak_abs,sample_name, batch_id, showplots):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_wavenb, baseline_abs, label = 'Baseline Subtracted Data')
    ax.plot(peak_wv, peak_abs, 'ro', label = 'peaks')
    for s, d in zip(peak_wv, peak_abs):
            plt.annotate(round(s, 2), xy = (s,d), rotation = 90)
    if batch_id is not None:
        ax.set_title(f'{sample_name} from batch_d {batch_id}')
    else:
        ax.set_title(f'{sample_name}')
    ax.set_xlabel('wavenumber ($cm^{-1}$)')
    ax.set_ylabel('absorbance')
    ax.legend()
    if showplots:
        plt.show()
    else:
        plt.close(fig)
    return fig

def baseline_correction_prospecpy_objects(list_of_propspecpy_object, showplot = False, save = True, verbose = True):
    """
    Batched adaptation of second_deriv function.
    """

    for prospecpy_obj in list_of_propspecpy_object:
        prospecpy_obj.subtract_baseline(save, showplot,verbose)