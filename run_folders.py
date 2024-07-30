import glob
from main import run, is_numeric
import os
import yaml

#------------PARAMETERS------------------------------------------------------------------
print('\n')
path = input('Path:\t')
print('\n')

file_path = os.path.join(path, "config.yaml")
if os.path.exists(file_path):
    print("Parameter file found in folder: "+os.path.basename(path))
    with open(os.path.join(path,"config.yaml"), 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
        frequency = params["frequency"]
        duration = params["duration"]
        threshold_freq = params["threshold_freq"]
        threshold_arrythmia = params["threshold_arrythmia"]

        show = params["show"]
        sigma0 = params["sigma0"]
        threshold_abs = params["threshold_abs"]
        prominence = params["prominence"]
        varfreq_weight = params["varfreq_weight"]
        scale = params["scale"]
else:
    print("No paramter file found. Specify the following paramters:\n")
    frequency = input('Pacing / Eigen frequency:\t')
    if is_numeric(frequency):
        frequency = float(frequency)
    print('\n')
    duration = float(input('Duration of signal in sec:\t'))
    print('\n')
    threshold_freq = float(input('Threshold for Tachycardia / Bradicardia in Hz:\t'))
    print('\n')
    threshold_arrythmia = float(input('Threshold for Arrythmia:\t'))
    print('\n')

    show = False  
    sigma0 = .01    
    threshold_abs = .1  
    prominence = 0.05 
    varfreq_weight = 0.5    
    scale = 1.  

    params = {
        'frequency': frequency,
        'duration': duration,
        'threshold_freq': threshold_freq,
        'threshold_arrythmia': threshold_arrythmia,
        'show': show,               	        # set to True if you want to check the plots and edit them.
        'sigma0': sigma0,                       # inital width of gaussian smoothing
        'threshold_abs': threshold_abs,         # threshold for peak detection of wavelet transform
        'prominence': prominence,               # omit any peaks in the correlation smaller than this value
        'varfreq_weight': varfreq_weight,       # weight of VF in weight sum of IMC and VF
        'scale': scale                          # scales the VF distribution before cropping to e [0,1]
    }
    with open(os.path.join(path,'config.yaml'), 'w') as yaml_file:
        yaml.dump(params, yaml_file)
#------------PARAMETERS------------------------------------------------------------------

directories = [ os.path.join(path,name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

files = []
for p in directories:
    files_single =  glob.glob(p+'\*')
    files += files_single

run(files, frequency = frequency, threshold_freq = threshold_freq, prominence = prominence, threshold_arrythmia = threshold_arrythmia, 
    duration = duration, sigma0 = sigma0, threshold_abs = threshold_abs, path = path, show = show)
