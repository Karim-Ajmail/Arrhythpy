# Arrhythpy
Arrhythpy is an open-source python based program to qunatify and classify arrhthmias in calcium transients in an automated manner.
See the corresponding paper at ... for explanation.
Please fell free to contact me at karim.ajmail@mtl.maxplanckschools.de if there are any questions or if you need help to get startet!

The easiest way to use Arrhythpy is to download the precomplied zip, that contains the executable. Please take to install the correct one for your operating system i.e. Arrhythpy_win for windows and Arrhythpy_ios for mac. This should run on its own without the need to install python itself.

If this does not work due to compatability issues, please follow the following instructions:
- Install python. The anaconda distribution is the easiest to do. Make sure that Anaconda prompt or Anaconda powershell prompt or any other python terminal is installed (should be installed as part of Anaconda).
- Download the 3 scripts run.py ; main.py and GUI.py and save them in the same folder.
- Open the terminal (e.g. anaconda prompt) and navigate to the folder, where the 3 scripts are located using the cd command i.e. type cd User/path/to/scripts
- Type the following commands to setup the environment
  - conda create -n arrhythpy
  - conda activate arrhythpy
  - conda install pip
  - pip install numpy
  - pip install matplotlib
  - pip install glob2
  - pip install pywavelet
  - pip install openpyxl
  - pip install tqdm
  - pip install scikit-learn
  - pip install scikit-image
  - pip install tk
  - pip install PyYAML
  - pip install tifffile
  - pip install scipy
  - pip install seaborn
  - pip install pyinstaller
  - pyinstaller -n Arrhythpy run.py
 
After finishing you should find the executable called Arrhthpy.exe in the dist folder created in the directory. Run it by double clicking and the Arrhthpy GUI should open.
    
