# Arrhythpy

**Arrhythpy** is an open-source Python-based program designed to quantify and classify arrhythmias in calcium transients in an automated manner. See the corresponding paper at [link to paper] for a detailed explanation.

For any questions or assistance, please feel free to contact me at [karim.ajmail@mtl.maxplanckschools.de](mailto:karim.ajmail@mtl.maxplanckschools.de).
So far, Arrhythpy only reads lsm files from line scans as described in the paper. If you need support for an other file type, let me know and I will add it.

## Installation

### Easiest Way

The easiest way to use Arrhythpy is to download the precompiled zip file that contains the executable. To this end, go to 'Releases' at Arrhythpy v1.0.0. Ensure you install the correct version for your operating system:
- **Windows:** `Arrhythpy.zip`
- **Mac:** `Arrhythpy_macOS.zip`

This executable should run independently without needing to install Python. Your operating system will probably warn you about the safety of the executable. Please trust me and download anyway (on windows: click 'More info' and the 'Run anyway')

### Alternative Installation

If the precompiled version does not work, you have to compile the executable yourself. To this end, downlaod the python scripts by cloning the repositry:
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/my-music-project.git
   cd Arrhythpy
2. Create new environment:
   ```sh
   conda create arrhythpy_env python==3.10
   conda activate arrhythpy_env
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
4. Generate the executable via pyinstaller:
   ```sh
   pyinstaller -n Arrhythpy --onefile run.py
