"""
@author: Piotr Bruzdziak
Department of Physical Chemistry
Gdansk University of Technology
e-mail: piotr.bruzdziak@pg.edu.pl

This script allows to fit and subtract vapor spectra from a series of raw 
measured spectra. Feel free to use or modify the script, but please cite the 
paper where the method has been described:
    
How to use:

obj_spec,corr_spectra = atm_subtraction(wavenb,spectra_to_subtract,atm_spectra)

The main fauntion to use is atm_subtraction. It employs AtmFitParams class 
to perform the fitting of a spectrum with vapor spectra. 

atm_subtraction arguments:
    wavenb          - wavenumbers corresponding to raw spectra
    spectra_to_subtract - numpy array of raw spectra (columnwise)
    atm_spectra     - numpy array of vapor spectra (columnwise)
    additional keywords:
      SG_poly=3,SG_points=21 - parameters of Savitzky-Golay smoothing algorithm
    
    wavenumbers, raw and vapor spectra must have the same number of rows!
    
This function returns:

corr_spectra        - numpy array of all vapor corrected spectra (columnwise)    
obj_spec            - a list of spectra objects with the following useful 
                      properties and methods:
    .fit_atm_params - subtraction coefficients for consecutive vapor spectra
                      (last three coefficients correspond to the baseline!)
    .sub_spectrum   - vapor-corrected spectrum
    .plot()         - plots raw and corrected spectra 
					 (saving plot to file keywords: save=True, 
													filename='figure_name.png')
        
    example:
        obj_spectra[0].fit_atm_params - will produce subtraction (and baseline)
                                        coeffcients of the first (0) spectrum

"""
import numpy as np
from scipy.signal import savgol_filter as savitzky_golay
from scipy.optimize import least_squares as leastsq
import matplotlib.pyplot as plt
from typing import AnyStr


def atm_subtraction(wavenb,spectra_to_subtract,atm_spectra,SG_poly=3,\
                    SG_points=21):
    spectra_to_subtract = np.asarray(spectra_to_subtract)
    list_of_spectra = []
    spectra_corrected = []
    if len(spectra_to_subtract.shape) == 1:
        no_of_spectra = 1
        list_of_spectra.append(AtmFitParams(wavenb,spectra_to_subtract,\
                                                atm_spectra,SG_poly=SG_poly,\
                                                SG_points=SG_points))
        spectra_corrected.append(list_of_spectra[0].sub_spectrum)
    else:
        no_of_spectra = spectra_to_subtract.shape[1]
        for i in range(no_of_spectra):
            list_of_spectra.append(AtmFitParams(wavenb,spectra_to_subtract[:,i],\
                                                atm_spectra,SG_poly=SG_poly,\
                                                SG_points=SG_points))
            spectra_corrected.append(list_of_spectra[i].sub_spectrum)
    return list_of_spectra,np.asarray(spectra_corrected).T
    

class AtmFitParams:    
    def __init__(self,wavenb,spectrum,atm_spectra,init_factor=10.0,\
                 SG_poly=5,SG_points=3):
        # list of wavenumbers
        self.wavenb = np.asarray(wavenb,dtype=float)
        # spectrum from which atmosphere (atm) spectra will be subtracted
        self.spectrum = np.asarray(spectrum,dtype=float)
        # array of atmosphere spectra
        self.atm_spectra = np.asarray(atm_spectra,dtype=float)
        if len(self.atm_spectra.shape) == 1:
            self.no_of_atm_spctr = 1
        else:
            self.no_of_atm_spctr = self.atm_spectra.shape[1]
        # initial multiplying factor for atm spectra
        self.init_factor = init_factor
        # initial quadratic baseline parameters
        bln_params = [0.0, 0.0, 0.0]
        # initial parameters of atm fitting
        self.init_fit_params = np.asarray((self.no_of_atm_spctr*\
                                           [self.init_factor] + bln_params),\
                                            dtype=float)
        no_of_parameters = np.size(self.init_fit_params)
        # boundaries for all parameters (can be changed)
        # set to +/- inf for atm intensities and +/-0.001 - 0.1 for baseline
        bln_bounds = [[-0.001,-0.01,-0.1],[0.001,0.01,0.1]]
        self.bounds = (no_of_parameters*[-np.inf],no_of_parameters*[np.inf])
        self.bounds[0][-3:] = bln_bounds[0]
        self.bounds[1][-3:] = bln_bounds[1]
        # Savitzky-Golay smoothing factors
        self.SG_poly = SG_poly
        self.SG_points = SG_points
        # Fitted parameters of water vapor subtraction (baseline - last 3)
        self.fit_atm_params = self.fit()
        # Vapor-corrected spectrum
        self.sub_spectrum = self.atm_subtract()   
    def baseline(self,bln_params):
        a = bln_params[0] #ax^2
        b = bln_params[1] #bx
        c = bln_params[2] #c
        return a*((self.wavenb)**2) + b*(self.wavenb) + c    
    def residuals(self,params,spectrum,wavenb,atm_spectra):
        baseln = self.baseline(params[-3:])
        if len(self.atm_spectra.shape) == 1:
            atm_sum = params[:-3]*atm_spectra
        else:
            atm_sum = np.sum((params[:-3]*atm_spectra),axis=1)
        total_sum = spectrum - atm_sum
        smoothed_total_sum = savitzky_golay(total_sum,self.SG_points,\
                                            self.SG_poly)
        return (smoothed_total_sum - total_sum + baseln)
    def fit(self):
        fit_result = leastsq(self.residuals,self.init_fit_params,args=\
                             (self.spectrum,self.wavenb,self.atm_spectra),\
                             ftol=1.49012e-10, xtol=1.49012e-10,\
                             bounds=self.bounds)
        fit_params = fit_result.x
        return fit_params
    def atm_subtract(self):
        baseln = self.baseline(self.fit_atm_params[-3:])
        if len(self.atm_spectra.shape) == 1:
            atm_sum = self.fit_atm_params[:-3]*self.atm_spectra
        else:
            atm_sum = np.sum((self.fit_atm_params[:-3]*self.atm_spectra),axis=1)
        sub_spectrum = self.spectrum - atm_sum - baseln
        return sub_spectrum
    def plot(self,sample_name = None, batch_id = None, showplots = False):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(self.wavenb,self.sub_spectrum,c='red',label='corrected spec.',\
                linewidth=0.5)
        ax.plot(self.wavenb,self.spectrum,c='black',label='original spec.',\
                linewidth=0.5)
        if batch_id is not None:
            ax.set_title(f'{sample_name} from batch_id {batch_id}')
        else:
            ax.set_title(sample_name)
        ax.legend()
        ax.set_xlabel('wavenumber ($cm^{-1}$)')
        ax.set_ylabel('absorbance')
        if showplots:
            plt.show()
        else:
            plt.close(fig)
        return fig

