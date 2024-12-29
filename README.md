# Arrhythpy

**Arrhythpy** is an open-source Python-based program designed to quantify and classify arrhythmias in calcium transients in an automated manner. See the corresponding paper at [link to paper] for a detailed explanation.

For any questions or assistance, please feel free to contact me at [karim.ajmail@mtl.maxplanckschools.de](mailto:karim.ajmail@mtl.maxplanckschools.de). So far, Arrhythpy only reads lsm or tif files from line scans as described in the paper. Additionally, there is the possibility to read in the transients directly as a csv file. If you need support for another file type, let me know and I will add it.

## Installation

### Easiest Way

The easiest way to use Arrhythpy is to download the precompiled zip file that contains the executable. To do this, go to 'Releases'. Ensure you install the correct version for your operating system:

- **Windows:** `Arrhythpy.zip` in the Arrhythpy for Windows release
- **Mac:** `Arrhythpy.zip` in the Arrhythpy for MacOS release

This executable should run independently without needing to install Python. Your operating system will probably warn you about the safety of the executable. Please trust me and download it anyway (on Windows: click 'More info' and then 'Run anyway').

### Alternative Installation

If the precompiled version does not work or your institution doesn't allow to download executables without official license, you will need to compile the executable yourself. To do this, first install python and download the Python scripts typing these commands into a terminal (e.g. Anaconda Prompt):

**1. Clone the repository:**
   ```sh
   git clone https://github.com/karim-ajmail/Arrhythpy.git
   cd Arrhythpy
   ```

**2. Create a new environment:**
   ```sh
   conda create -n arrhythpy_env python=3.10
   conda activate arrhythpy_env
   ```

**3. Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

**4. Generate the executable via PyInstaller:**
   ```sh
   pyinstaller -n Arrhythpy --onefile run.py
   ```
When you navigate to the 'dist' folder, you will find the executable called Arrhythpy. You can move to a convenient directory if you like and can be easily run to open the GUI.

## Running
Once installed, simply double-click to run. It takes some time until the program is open, but eventually the GUI window is opened. Please refer to the Manual for information on the parameters. After running the programm the results are saved as an excel file including some figures about the data.

