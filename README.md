# Arrhythpy

**Arrhythpy** is an open-source Python-based program designed to quantify and classify arrhythmias in calcium transients in an automated manner. See the corresponding paper at [link to paper] for a detailed explanation.

For any questions or assistance, please feel free to contact me at [karim.ajmail@mtl.maxplanckschools.de](mailto:karim.ajmail@mtl.maxplanckschools.de). So far, Arrhythpy only reads lsm files from line scans as described in the paper. If you need support for another file type, let me know and I will add it.

## Installation

### Easiest Way

The easiest way to use Arrhythpy is to download the precompiled zip file that contains the executable. To do this, go to 'Releases' at Arrhythpy v1.0.0. Ensure you install the correct version for your operating system:

- **Windows:** `Arrhythpy.zip`
- **Mac:** `Arrhythpy_macOS.zip`

This executable should run independently without needing to install Python. Your operating system will probably warn you about the safety of the executable. Please trust me and download it anyway (on Windows: click 'More info' and then 'Run anyway').

### Alternative Installation

If the precompiled version does not work, you will need to compile the executable yourself. To do this, download the Python scripts by cloning the repository:

**1. Clone the repository:**
   ```sh
   git clone https://github.com/Karim-Ajmail/Arrhythpy.git
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
