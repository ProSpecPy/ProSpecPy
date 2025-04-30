from __future__ import annotations

import numpy as np
import pandas as pd

# find peaks
from scipy.signal import find_peaks, peak_widths

# widget


def get_peaks(second_deriv, threshold=0.15):  # , showplot = False):
    """
    Function to detect peaks in the second derivative of a spline function.

    Parameters:
    - second_deriv: tuple
        Tuple containing the results of the spline function, including x and y values.
    - threshold: float, optional (default=0.15)
        Threshold value used to determine the prominence of peaks.
    - showplot: bool, optional (default=False)
        Flag to indicate whether to plot the detected peaks.

    Returns:
    - peaks_index: array-like
        Indices of the detected peaks.
    - deriv_x_peak_val: array-like
        x-coordinate values(wavenumbers) of the detected peaks.
    """
    d2ydx2_spl_upsidedown = second_deriv[1] * -1
    relative_height = threshold * max(d2ydx2_spl_upsidedown)
    peaks_index = find_peaks(d2ydx2_spl_upsidedown, prominence=relative_height)

    # use for loops to extract the coordinates of the peaks so we can plot them on the plot above
    d2ydx2_peak_val = []
    deriv_x_peak_val = []

    for i in peaks_index[0]:
        d2ydx2_peak = second_deriv[1][i]
        deriv_x_peak = second_deriv[2][i]

        d2ydx2_peak_val.append(d2ydx2_peak)
        deriv_x_peak_val.append(deriv_x_peak)
    return peaks_index, deriv_x_peak_val, d2ydx2_peak_val


def get_start_end_anchorpoints(peaks_index, second_deriv):
    """
    Function to determine the start and end anchor points of peaks.

    Parameters:
    - peaks_index: array-like
        Indices of the detected peaks.
    - second_deriv: tuple
        Second derivative of the spline function

    Returns:
    - wv_startIdx: list
        x-coordinate values(wavenumber) of the start anchor points of peaks.
    - wv_endIdx: list
        x-coordinate values(wavenumber) of the end anchor points of peaks.
    """
    d2ydx2_spl_upsidedown = second_deriv[1] * -1
    peak_wid = peak_widths(d2ydx2_spl_upsidedown, peaks_index, rel_height=1)

    width_endIdx = [int(x) for x in peak_wid[2]]
    wv_endIdx = []

    for i in width_endIdx:
        wv_endIdx.append(second_deriv[2][i])

    width_startIdx = [int(x) for x in peak_wid[3]]
    wv_startIdx = []

    for i in width_startIdx:
        wv_startIdx.append(second_deriv[2][i])
    return wv_startIdx, wv_endIdx


def get_all_anchor_points(
    wv_startIdx, wv_endIdx, deriv_x_peak_val, anchor_points_raw_data, y_corr_abs, adj_factor=1
):
    """
    Function to filter and post-process anchor points based on peak characteristics.

    Parameters:
    - wv_startIdx: list
        x-coordinate values (wavenumber) of the start anchor points of peaks.
    - wv_endIdx: list
        x-coordinate values (wavenumber) of the end anchor points of peaks.
    - deriv_x_peak_val: array-like
        x-coordinate values(wavenumber) of the detected peaks.
    - anchor_points_raw_data: array-like
        Raw spectra data.
    - y_corr_abs: array-like
        Corresponding absorbance values.
    - plot_title: str, optional (default=None)
        Title for the plot.
    - adj_factor: float, optional (default=1)
        Adjustment factor for filtering anchor points.
    - show_plot: bool, optional (default=True)
        Flag to indicate whether to display the plot.

    Returns:
    - anchor_data_sorted: DataFrame
        DataFrame containing sorted anchor points data.
    """
    # get the smaller width for each peak using get_smaller_peak_width()
    smaller_peak_wid = get_smaller_peak_width(deriv_x_peak_val, wv_startIdx, wv_endIdx)
    post_process_anchor_points = []
    post_process_anchor_points_abs = []
    """
    Algorithm:
        For each point in the raw data:
            1. get the index of the peak that is closest to this point using the criteria minimum(abs(peak_wavenumber - raw_point))
            2. if abs(closest peak - raw data point) > smaller peak width * adjustment factor:
                        then append the raw data point as anchor point and its absorbance
    """

    for index in range(len(anchor_points_raw_data)):
        dist_peak_to_anchor = abs(deriv_x_peak_val - anchor_points_raw_data[index])
        closest_peak_idx = np.argmin(dist_peak_to_anchor)

        if (
            abs(deriv_x_peak_val[closest_peak_idx] - anchor_points_raw_data[index])
            > smaller_peak_wid[closest_peak_idx] * adj_factor
        ):
            post_process_anchor_points.append(anchor_points_raw_data[index])
            post_process_anchor_points_abs.append(y_corr_abs[index])

    # post processesing to avoid repeating values and make sure the wavenumbers are in the same acending or decending order
    post_process_anchor_data = pd.DataFrame(
        {"wavenumber": post_process_anchor_points, "absorbance": post_process_anchor_points_abs}
    )
    post_process_anchor_data = post_process_anchor_data.drop_duplicates()
    anchor_data_sorted = post_process_anchor_data.sort_values(by="wavenumber").reset_index()

    # get all peak wavenumber and absorbance for plotting
    peak_wavenumber, peak_absorbance = get_peaks_absorbance(
        deriv_x_peak_val, anchor_points_raw_data, y_corr_abs
    )
    return anchor_data_sorted, peak_wavenumber, peak_absorbance


def get_peaks_absorbance(deriv_x_peak_val, x_wavenb, y_corr_abs):
    """
    Function to retrieve peak wavenumbers and corresponding absorbance values.

    Parameters:
    - deriv_x_peak_val: array-like
        x-coordinate values (wavenumber) of the detected peaks.
    - x_wavenb: array-like
        Wavenumber values of raw spectra.
    - y_corr_abs: array-like
        Corresponding absorbance values of raw spectra.

    Returns:
    - peak_wavenumber: list
        Wavenumber values corresponding to the peaks.
    - peak_absorbance: list
        Absorbance values corresponding to the peaks.
    """
    # Define the range
    range_width = 2
    peak_wavenumber = []
    peak_absorbance = []

    for peak_val in deriv_x_peak_val:
        indices_within_threshold = [
            index for index, value in enumerate(x_wavenb) if abs(value - peak_val) <= range_width
        ]
        data = pd.DataFrame(
            {"wv": x_wavenb[indices_within_threshold], "abs": y_corr_abs[indices_within_threshold]}
        )
        # Choosing the highest wavenumber as the peak
        peak_data = data.loc[data["wv"].idxmax()]
        peak_wavenumber.append(peak_data["wv"])
        peak_absorbance.append(peak_data["abs"])
    return peak_wavenumber, peak_absorbance


def get_smaller_peak_width(deriv_x_peak_val, wv_startIdx, wv_endIdx):
    """
    Function to calculate the smaller peak width for each peak.

    Parameters:
    - deriv_x_peak_val: array-like
        x-coordinate values (wavenumber) of the detected peaks.
    - wv_startIdx: list
        x-coordinate values (wavenumber) of the start anchor points of peaks.
    - wv_endIdx: list
        x-coordinate values (wavenumber) of the end anchor points of peaks.

    Returns:
    - smaller_peak_wid: list
        Smaller peak width for each peak.
    """
    smaller_peak_wid = []
    for i in range(len(deriv_x_peak_val)):
        left_wid = deriv_x_peak_val[i] - wv_startIdx[i]
        right_wid = wv_endIdx[i] - deriv_x_peak_val[i]
        # smaller peak width
        smaller_peak_wid.append(min(left_wid, right_wid))
    return smaller_peak_wid


def get_peak_wid_at_half_height(baseline_corrected_abs, peak_wv_index):
    peak_wid = peak_widths(baseline_corrected_abs, peak_wv_index, rel_height=0.5)
    return peak_wid[0]
