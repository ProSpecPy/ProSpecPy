from scipy.optimize import curve_fit
from scipy.signal import peak_widths
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, *params):

    y = np.zeros_like(x)

    for i in range(0, len(params), 3):
        amplitude = params[i]
        center = params[i+1]
        sigma = params[i+2]
        y += amplitude*(1/(sigma* np.sqrt(2*np.pi)))*np.exp((-1/2)*((x - center)/ sigma)**2) 
    return y

def lorentzian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude = params[i]
        center = params[i+1]
        sigma = params[i+2]
        y += amplitude*sigma**2/((x-center)**2+sigma**2)
    return y


def peak_fit(fit_function, x_wavenumber, y_absorbance, peak_index, sample_name, batch_id,showplot = True): 

    get_peak_wid = peak_widths(y_absorbance, peak_index, rel_height=1) 
    all_peak_wid = get_peak_wid[0]
    guess = []
    
    for i, idx in enumerate(peak_index):
        peak_height = y_absorbance[idx]
        guess.append(peak_height)
        peak_center = x_wavenumber[idx]
        guess.append(peak_center)
        peak_wid = all_peak_wid[i]
        guess.append(peak_wid)

    if fit_function == 'Gaussian':
        objective_function = gaussian
    elif fit_function == 'Lorentzian':
        objective_function = lorentzian
    else:
        raise ValueError("fit_function must be 'Gaussian' or 'Lorentzian'")  # Handle invalid input

    params, _ = curve_fit(objective_function, x_wavenumber, y_absorbance, guess)

    # Calculate RMSE
    predicted_absorbance = objective_function(x_wavenumber, *params)
    residuals = y_absorbance - predicted_absorbance
    rmse = np.sqrt(np.mean(residuals**2))
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_wavenumber,y_absorbance, label = 'baseline corrected data')
    ax.plot(x_wavenumber, predicted_absorbance, label = f'{fit_function} curve fit')
    if batch_id is not None:
        ax.set_title(f'{sample_name} from batch_d {batch_id}')
    else:
        ax.set_title(f'{sample_name}')
    ax.set_xlabel('wavenumber ($cm^{-1}$)')
    ax.set_ylabel('absorbance')
    ax.legend()


    if showplot:
        plt.show()
    else:
        plt.close(fig)
    return params, rmse, fig


def gaussian_fit_prospecpy_objects(list_of_propspecpy_object, show_plots = False, save = True, verbose = True):
    """
    Batched adaptation of gaussian_fit_baseline function.
    """

    for prospecpy_obj in list_of_propspecpy_object:
            #print(f'processing plots for {prospecpy_obj.sample_name}')
            prospecpy_obj.gaussian_fit_baseline(show_plots, save, verbose)

def lorentzian_fit_prospecpy_objects(list_of_propspecpy_object, show_plots = False, save = True, verbose = True):
    """
    Batched adaptation of lorentzian_fit_baseline function.
    """

    for prospecpy_obj in list_of_propspecpy_object:
            #print(f'processing plots for {prospecpy_obj.sample_name}')
            prospecpy_obj.lorentzian_fit_baseline(show_plots, save, verbose)

