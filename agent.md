\# Project Name: Pluto - Adaptive 3D Image Segmentation



\## Context

I have a Python script called `pluto.py` that I wrote for personal use. It features a custom adaptive segmentation algorithm for 3D image stacks, based on the assumption that gray values do not change drastically between consecutive 2D slices. 



Currently, the script lacks a graphical interface, making it difficult for other users to operate, and it crashes by throwing an error whenever an empty mask is encountered.



\## Goal

I need a Python application that uses my exact custom segmentation logic but wraps it in a minimal, interactive desktop application with proper error handling.



\## Core Requirements

1\. \*\*Data Loading:\*\* 

&#x20;  - Load a stack of 2D images (usually \*\*TIFF files\*\*) from `./data/{folder\_name}`.

&#x20;  - The images are ordered numerically by filename.

2\. \*\*User Interface:\*\* 

&#x20;  - Build a \*\*minimal desktop window\*\* (e.g., using Tkinter + OpenCV or OpenCV's native GUI).

&#x20;  - Allow the user to scroll/scrub through the image stack.

&#x20;  - Let the user decide which image to start on and draw the initial mask by hand.

3\. \*\*Algorithm Execution:\*\* 

&#x20;  - Once the user manually segments the starting image, apply the `pluto.py` logic to propagate the mask to the next consecutive images.

4\. \*\*Empty Mask Handling:\*\* 

&#x20;  - If the algorithm generates an empty mask (e.g., 0 segmented pixels), \*\*pause\*\* the execution.

&#x20;  - Notify the user via the UI.

&#x20;  - Give the user two options: \*\*Stop execution\*\* OR \*\*Segment by hand\*\*.

&#x20;  - If the user decides to segment the image by hand, resume the automatic segmentation sequence starting from that newly drawn mask.

5\. \*\*Saving:\*\* 

&#x20;  - Upon completion, or if the user chooses to stop at an empty mask, save all generated masks into `./masks/{folder\_name}`.

&#x20;  - All masks must be strictly saved as \*\*binary PNG files\*\*.



\## Original Logic (`pluto.py`)

\[INSERT YOUR PLUTO.PY CODE HERE OR EXPLAIN THE EXACT MATH/LOGIC]



\## Directory Structure

Please generate the codebase strictly following this project structure:



Pluto/

&#x20; ├── requirements.txt

&#x20; ├── main.py

&#x20; ├── Utils/

&#x20; │   └── utils.py

&#x20; ├── data/

&#x20; └── masks/

