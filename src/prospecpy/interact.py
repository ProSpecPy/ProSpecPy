
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
import csv # to export the anchor point coordinates

def interact(prospecpy_objects, threshold_guess, adj_guess):
    #First object in the interactive session is the file name selection
    #create a mapping between sample names and its respective prospecypy objects
    sampleName_prospecpyObj_map = {}
    for prospecpy_obj in prospecpy_objects:
        sampleName_prospecpyObj_map[prospecpy_obj.sample_name] = prospecpy_obj

    sorted_file_names = sorted(sampleName_prospecpyObj_map.keys())

    file_widget = widgets.ToggleButtons(
                    options = sorted_file_names,
                    description='Step 1. File selection:',
                    disabled=False
                    )
    style = {'description_width': '500px'}
    #Preset threshold widget range and step by us
    threshold_widget = widgets.BoundedFloatText(
        value=threshold_guess,
        min=0,
        max=1,
        step=0.01,
        description='Threshold for peak selection(0 to 1 in 0.01 steps):',
        disabled=False,
        layout=widgets.Layout(width='70%'),
        style = style
    )

    #Preset adj widget range and step by us
    adj_widget = widgets.BoundedFloatText(
        value=adj_guess,
        min=0,
        max=5,
        step=0.01,
        description='adj for anchor point selection(0 to 5 in 0.01 steps):',
        disabled=False,
        layout=widgets.Layout(width='70%'),
        style=style
    )

    """baseline_widget = widgets.BoundedFloatText(
        value=threshold_guess,
        min=0,
        max=1,
        step=0.001,
        description='Smooth factor for baseline curve fitting:',
        disabled=False,
        layout=widgets.Layout(width='70%'),
        style = style
    )"""

    Save = widgets.Button(
    description ='Save'
    )

    #undo button is deleted after deciding for save bottom to overwrite previous values
    #Undo = widgets.Button(
    #description = 'Undo'
    #)

    threshold_save = []
    adj_save = []
    file_save = [] 
    #anchor_point_save = {} #potiential: if we want to output anchor points of different samples together

    #location of def botton_click is shifted down such that the anchor point values can be saved
    
    #def do_over(b):
    #    try:
    #        del file_save[-1]
    #        del threshold_save[-1]
    #        del adj_save[-1]
    #    except IndexError:
    #        print("Cannot delete values. No saved values for threshold and adjustment factor found. Please submit values before Oops")



    
    def interact_with_functions(sample_name, threshold, adj):
        
        #plotting subtracted spectra
        prospecpy_obj = sampleName_prospecpyObj_map[sample_name]
        prospecpy_obj.plot_subtracted_spectra(save = False, showplots = True)

        prospecpy_obj.peak_finder(threshold)
        peak_info_dict = prospecpy_obj.get_second_deriv_peak_dict()
        second_deriv_dict = prospecpy_obj.get_second_deriv_dict()

        plt.figure(figsize=(18, 6))

        #second derivative plot
        plt.subplot(1,2,1)

        plt.plot(peak_info_dict['peak_wavenumber'], peak_info_dict['peak_second_deriv_absorbance'], "ro",label = "peak finder peaks")
        plt.plot(second_deriv_dict['wavenumber'], second_deriv_dict['absorbance'], label = "spline results")
        plt.title("Second derivative plot peak selection")
        plt.xlabel("wavenumber ($cm^{-1}$)")
        plt.ylabel("second derivative (absorbance)")
        plt.legend()

        #anchor points curve fitting
        prospecpy_obj.anchor_point_fit(adj)
        anchor_points_peak_dict = prospecpy_obj.get_anchor_points_peak_dict()
        anchor_point_peak_wv = anchor_points_peak_dict['peak_wavenumber']
        anchor_point_peak_absorbance = anchor_points_peak_dict['peak_absorbance']
        anchor_points_data = prospecpy_obj.get_anchor_points()
        anchor_points_wv = anchor_points_data['wavenumber']
        anchor_points_abs = anchor_points_data['absorbance']

        prospecpy_obj.baseline_fit()
        baseline_curve = prospecpy_obj.get_baseline_curve()

        plt.subplot(1,2,2)
        plt.plot(prospecpy_obj.get_subtracted_spectra_wavenumber(), prospecpy_obj.get_subtracted_spectra_absorbance())
        plt.plot(anchor_point_peak_wv,anchor_point_peak_absorbance,'ro', label='peaks')
        plt.plot(anchor_points_wv, anchor_points_abs, 'bx', label = 'anchor_points')
        plt.plot(baseline_curve['wavenumber'], baseline_curve['absorbance'], 'g--', label = 'baseline fit')
        plt.xlabel("wavenumber")
        plt.ylabel("Absorbance")
        plt.title("Anchor point selection")
        plt.legend()

        plt.tight_layout()

        print('Step 2. Use the threshold and adj widgets to adjust the number of peaks included and the where to put the anchor points, and click submit after desired outcome to save the final parameters')
        
        #location of button_click function shifted to save the anchor point coordinates
        def button_click(a):
            #global threshold_save
            #global adj_save
            #global file_name_save
            threshold_save.append(threshold_widget.value)
            adj_save.append(adj_widget.value)
            file_save.append(file_widget.value)
            
            #prepare anchor point coordinates
            anchor_point_wv_save = anchor_points_wv
            anchor_point_ab_save = anchor_points_abs
            rows = zip(anchor_point_wv_save, anchor_point_ab_save)
            
            #export the samples in their individual csv file
            file_path = f'{prospecpy_obj.output_folder}/anchor_point_coordinates.csv'

            with open(file_path, mode='w', newline='') as file:
                #create a csv writer object
                writer = csv.writer(file)
                
                #write the headers
                headers = ["Wavenumber", "Asborbance"]
                writer.writerow(headers)

                #wirte the data
                writer.writerows(rows)
            
            return file_save, threshold_save, adj_save

        Save.on_click(button_click)
        #display(submit)

        #Undo.on_click(do_over)
      
    #use one output because the output has to follow structure of ipywidget output and only interactive and produce non package specific objects
    interactive_results = widgets.interactive(interact_with_functions, sample_name = file_widget, threshold = threshold_widget, adj = adj_widget)
    
    #print(interactive_results)
    file_selection_display = interactive_results.children[0]
    threshold_adj_plot_display = interactive_results.children[1]
    threshold_adj_display = interactive_results.children[2]
    raw_display = interactive_results.children[3]

    display(file_selection_display)
    display(raw_display) 
    display(threshold_adj_plot_display)
    display(threshold_adj_display)
    
    display(Save)
    #display(Undo)



