# Computer-Vision-CW-COMP3065-UNNC-2024-5-6th

**Task:** Panorama generation from videos

**Author:** Zepei Luo, 20321548, scyzl9@nottingham.edu.cn

## Key Features & Functionalities

1. Overcome the excessive distortion problem of conventional methods.
2. Smooth the chromatic aberration due to exposure problems (Feathering).
3. Handle videos with persistent shaking.
4. Design UI to visualize the panorama-generating process and the above functionalities.

## Project Structure

- **main_new.py:** The main file that needs to be run, contains codes of UI, and calling function in `gen_panorama.py`.
- **gen_panorama.py:** Contains the functions to implement panorama generation.
- **utils.py:** Contains some utility functions.

## Installation

Before running the code, install the necessary packages:

```bash
pip install opencv-python
pip install imutils

## Run

Execute the following command:
```bash
python main_new.py
