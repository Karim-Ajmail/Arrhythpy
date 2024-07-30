import tkinter as tk
from tkinter import ttk, filedialog,messagebox
import glob
import os
import numpy as np
import yaml
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from tifffile import imread
import openpyxl
import pywt 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d,gaussian_filter
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score
import seaborn as sns
import yaml
from main import run,analyse_single,load_transient,calc_accuracy,get_threshold_freq,get_threshold_arrythmia
from GUI import ArrhythpyGUI

if __name__ == "__main__":
    app = ArrhythpyGUI()
    app.mainloop()
