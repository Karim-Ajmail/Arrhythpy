import tkinter as tk
from tkinter import ttk, filedialog,messagebox
import glob
import os
import numpy as np
import yaml
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from main import run,analyse_single,load_transient,calc_accuracy,get_threshold_freq,get_threshold_arrythmia
import threading
'''
GUI for Ahrrythpy. WRITTEN by Karim Ajmail 2023
'''
class ArrhythpyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Arrhythpy")
        self.parameters_visible = False
        self.additional_parameters_entries = []
        self.additional_parameters_labels = []
        self.files = []  # List of files to process
        self.classification = None
        self.accuracy = 0.
        self.stop = False
        self.imporved = False
        self.wait_var = tk.BooleanVar()
        self.create_widgets()

    def browse_path(self):
        '''
        load path to data
        '''
        path = filedialog.askdirectory()
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)
            self.files = glob.glob(path+'\*')

    def browse_file(self):
        '''
        load config file (optional)
        '''
        path = filedialog.askopenfilename()
        if path:
            if os.path.exists(path):
                with open(path, 'r') as yaml_file:
                    params = yaml.safe_load(yaml_file)
                    self.threshold_arrhythmia_entry.delete(0, tk.END)
                    self.threshold_arrhythmia_entry.insert(0, params["threshold_arrythmia"])
                    self.threshold_freq_entry.delete(0, tk.END)
                    self.threshold_freq_entry.insert(0,params["threshold_freq"])
                    self.duration_entry.delete(0, tk.END)
                    self.duration_entry.insert(0,params["duration"])
                    self.frequency_entry.delete(0, tk.END)
                    self.frequency_entry.insert(0,params["frequency"])
                    
                    parameters = {'Sigma0': params["sigma0"],'Prominence':params["prominence"],'Threshold Abs': params["threshold_abs"],'VarFreq Weight': params["varfreq_weight"],'Scale': params["scale"]}
                    for i, param_name in enumerate(parameters):
                        label = ttk.Label(self, text=param_name)
                        entry = ttk.Entry(self, width=30)
                        entry.insert(0, parameters[param_name])  # Set default value
                        label.grid(row=6+i, column=0, padx=5, pady=5, sticky=tk.E)
                        entry.grid(row=6+i, column=1, padx=5, pady=5, sticky=tk.W)
                        self.additional_parameters_entries.append(entry)
                        self.additional_parameters_labels.append(label)

    def toggle_parameters(self):
        if self.parameters_visible:
            # Hide additional parameters and adjust row configuration
            for widget in self.additional_parameters:
                widget.grid_remove()
            self.rowconfigure(6, weight=0)
            self.parameters_visible = False

        else:
            # Show additional parameters and adjust row configuration
            for widget in self.additional_parameters:
                widget.grid()
            self.rowconfigure(6, weight=1)
            self.parameters_visible = True

    def label_transients(self):
        '''
        open window to manually label transients for automated deterination of the two thresholds 
        '''
        frequency = float(self.frequency_entry.get())
        duration = float(self.duration_entry.get())
        additional_parameters = [float(entry.get()) for entry in self.additional_parameters_entries]
        threshold_arrhythmia = float(self.threshold_arrhythmia_entry.get())
        threshold_freq = float(self.threshold_freq_entry.get())
        files_shuffeled = random.sample(self.files, len(self.files))     

        res_temp = np.zeros((len(self.files),4))
        self.i = 0
        self.improved = False
        self.stop = False

        plot_window = tk.Toplevel(self)
        plot_window.geometry("750x580")  # Adjust the width and height as needed
        plot_window.title("Classify Transient!")
        done = ttk.Button(plot_window, text="Done", command= lambda: (setattr(self, 'stop', True), plot_window.destroy()))
        done.pack(side=tk.RIGHT, anchor=tk.SE)

        legend = ttk.Label(plot_window, text="\nClassification:")
        legend.place(x=300, y=500)
        legend_desc = ttk.Label(plot_window, text="1: Rhythmic | 2: Tachycardic Rhythmic | 3: Bradycardic Rhythmic | 4: Arrhythmic | 5: Tachycardic Arrhythmic | 6: Bradycardic Arrhythmic\n")
        legend_desc.place(x=5, y=520)

        classification_entry = ttk.Entry(plot_window)
        classification_entry.place(x=300, y=540)
        classification_entry.focus_set()  # Focus on the entry field
        classification_entry.bind("<Return>", lambda event: self.check_classification(classification_entry.get()))
        plot_window.bind("<Return>", lambda event: self.check_classification(classification_entry.get()))
        for file in files_shuffeled:
            intensity = load_transient(file)  
            if not np.isnan(intensity).any():
                self.show_plot(intensity,plot_window)
                arrythmia, tachcardia, bradycardia = analyse_single(intensity=intensity,frequency=frequency,
                                                        duration=duration, sigma0=additional_parameters[0], 
                                                        prominence=additional_parameters[1],
                                                        threshold_abs=additional_parameters[2],
                                                        varfreq_weight=additional_parameters[3],
                                                        scale=additional_parameters[4])
                res_temp[self.i] = arrythmia, tachcardia, bradycardia, self.classification
                self.i += 1
                self.results = res_temp[:self.i]
                try:
                    self.accuracy = calc_accuracy(self.results[:,0],self.results[:,1],self.results[:,2], self.results[:,3],threshold_arrhythmia, threshold_freq)
                    thresh_opt_freq,_ = get_threshold_freq(self.results[:,3],self.results[:,1],self.results[:,2])
                    thresh_opt_arr,_ = get_threshold_arrythmia(self.results[:,3],self.results[:,0])
                    accuracy_test = calc_accuracy(self.results[:,0],self.results[:,1],self.results[:,2], self.results[:,3],thresh_opt_arr, thresh_opt_freq)
                    print(f'\nAccuracy with default thresholds ({np.round(threshold_arrhythmia,2)} & {np.round(threshold_freq,2)}): {np.round(self.accuracy*100,2)} \nAccuracy with auto thresholds ({np.round(thresh_opt_arr,2)} & {np.round(thresh_opt_freq,2)}): {np.round(accuracy_test*100,2)}\n')

                    if accuracy_test > self.accuracy:
                        self.improved = True
                        self.accuracy = accuracy_test
                        self.threshold_arrhythmia_entry.delete(0, tk.END)
                        self.threshold_arrhythmia_entry.insert(0, thresh_opt_arr)
                        self.threshold_freq_entry.delete(0, tk.END)
                        self.threshold_freq_entry.insert(0,thresh_opt_freq)
                except:
                    print('Not enough labelled to fit. Need more labelled data!')

                if self.stop:
                    break
        if self.improved:
            print('Found optimal threshold successfully!')
        else:
            print('Optimal thresholds did not perform better than default. Sticking with default values.')
        messagebox.showinfo("Done classfying.", "All files are classified. You can now proceed to run Arrhythpy.")


    def show_plot(self,intensity,plot_window): 
        plt.close()
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0,float(self.duration_entry.get()),len(intensity)),intensity)  # Assuming data is a list of intensity values
        ax.set_xlabel("Time")
        ax.set_ylabel("Intensity")
        ax.set_title(f'Transient {self.i+1}                                 Accuracy: {np.round(self.accuracy*100,2)} %')
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=plot_window)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=50, y=10)
        self.wait_variable(self.wait_var)

    def check_classification(self, value):
        if value.isdigit() and int(value) in range(1, 7):  # Check if the entered value is a valid classification
            self.classification = int(value)
            
            self.wait_var.set(True)
        else:
            messagebox.showwarning("Invalid Classification", "Please enter a valid classification (1 to 6).")

    def run_wrapper(self):
        self.run_button.config(state='disabled')
        self.auto_thres.config(state='disabled')
        self.browse_button.config(state='disabled')
        self.browse_button_config.config(state='disabled')
        print('\nArrhythpy is running ...\n')
        # Access all the input values including the additional parameters
        path = self.path_entry.get()
        frequency = float(self.frequency_entry.get())
        duration = float(self.duration_entry.get())
        threshold_arrhythmia = float(self.threshold_arrhythmia_entry.get())
        threshold_freq = float(self.threshold_freq_entry.get())

        # Extract additional parameters from the additional entry widgets
        additional_parameters = [float(entry.get()) for entry in self.additional_parameters_entries]

        params = {
            'frequency': frequency,
            'duration': duration,
            'threshold_freq': threshold_freq,
            'threshold_arrythmia': threshold_arrhythmia,
            'sigma0': additional_parameters[0],
            'prominence': additional_parameters[1],
            'threshold_abs': additional_parameters[2],
            'varfreq_weight': additional_parameters[3],
            'scale': additional_parameters[4]
        }

        with open(os.path.join(path,'config.yaml'), 'w') as yaml_file:
            yaml.dump(params, yaml_file)

        files = glob.glob(path+'\*')
        # Call the run function with the extracted parameters
        run(files=files, frequency=frequency, threshold_freq=threshold_freq, threshold_arrythmia=threshold_arrhythmia,
                duration=duration, path=None, show=False,
                sigma0=additional_parameters[0], prominence=additional_parameters[1],
                threshold_abs=additional_parameters[2], varfreq_weight=additional_parameters[3],
                scale=additional_parameters[4],save_mode=additional_parameters[5])

        self.run_button.config(state='normal')
        self.auto_thres.config(state='normal')
        self.browse_button.config(state='normal')
        self.browse_button_config.config(state='normal')

    def run_with_progressbar(self):

        loading_window = tk.Toplevel(self)
        loading_window.title("Arrhythpy Running...")

        # Adjusting the size of the loading wheel
        style = ttk.Style(loading_window)
        style.configure('TProgressbar', thickness=10)

        progressbar = ttk.Progressbar(loading_window,length=250, mode='indeterminate',takefocus=True)
        progressbar.pack(padx=20, pady=10)

        progressbar.start()

        def run_function_and_close_loading():
            self.run_wrapper()
            loading_window.destroy()

        # Run the long-running function in a separate thread
        thread = threading.Thread(target=run_function_and_close_loading)
        thread.start()

    def create_widgets(self):
        # Create labels
        self.path_label = ttk.Label(self, text="Path:")
        self.frequency_label = ttk.Label(self, text="Frequency \ Hz:")
        self.duration_label = ttk.Label(self, text="Duration \ s:")
        self.threshold_arrhythmia_label = ttk.Label(self, text="Threshold Arrhythmia:")
        self.threshold_freq_label = ttk.Label(self, text="Threshold Frequency:")

        # Create entry widgets
        self.path_entry = ttk.Entry(self, width=30)
        self.frequency_entry = ttk.Entry(self, width=30)
        self.duration_entry = ttk.Entry(self, width=30)
        self.threshold_arrhythmia_entry = ttk.Entry(self, width=30)
        self.threshold_freq_entry = ttk.Entry(self, width=30)

        self.threshold_arrhythmia_entry.insert(0, 0.1)
        self.threshold_freq_entry.insert(0, 0.1)

        parameters = {'Sigma0': .01, 'Prominence': 0.05, 'Threshold Abs': .1, 'VarFreq Weight': 0.5, 'Scale': 1., 'Safe Mode':True}

        for i, param_name in enumerate(parameters):
            label = ttk.Label(self, text=param_name)
            entry = ttk.Entry(self, width=30)
            entry.insert(0, parameters[param_name])
            label.grid(row=6+i, column=0, padx=5, pady=5, sticky=tk.E)
            entry.grid(row=6+i, column=1, padx=5, pady=5, sticky=tk.W)
            self.additional_parameters_entries.append(entry)
            self.additional_parameters_labels.append(label)

        self.additional_parameters = self.additional_parameters_labels + self.additional_parameters_entries

        # Create buttons
        self.edit_parameters_button = ttk.Button(self, text="Edit Parameters", command=self.toggle_parameters)
        self.run_button = ttk.Button(self, text="RUN", command=self.run_wrapper)
        self.browse_button = ttk.Button(self, text="Search..", command=self.browse_path)
        self.browse_button_config = ttk.Button(self, text="Load config..", command=self.browse_file)
        self.auto_thres = ttk.Button(self, text="Find thresholds", command=self.label_transients)

        self.path_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
        self.path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        self.frequency_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
        self.frequency_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        self.duration_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)
        self.duration_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        self.threshold_arrhythmia_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.E)
        self.threshold_arrhythmia_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        self.threshold_freq_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.E)
        self.threshold_freq_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

        self.edit_parameters_button.grid(row=5, column=0, columnspan=2, pady=10)
        self.run_button.grid(row=5, column=2, columnspan=2, pady=10)
        self.browse_button_config.grid(row=3, column=2, columnspan=2, pady=10)
        self.auto_thres.grid(row=4, column=2, columnspan=2, pady=10)

        for widget in self.additional_parameters:
            widget.grid_remove()
        self.rowconfigure(6, weight=0)

if __name__ == "__main__":
    app = ArrhythpyGUI()
    app.mainloop()

