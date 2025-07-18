# CT Viewer Specifications
I want to create an CT viewer for my machine learning project using ipywidgets. It will enable me to visualize raw and processed data to validate that my image processing code is working as intended. Please read the requirements below and create a class with a method called display() which renders the UI.  

## Packages
Below is a list of packages that will be needed for some key features.
- ipywidgets
- matplotlib
- cv2
- nrrd

## Data Specification
There is one folder that will contain multiple folders, each of which contain the data for a single CT exam. Inside of each subfolder will be:
- The original 3D CT data (nrrd file)
- The original 3D segmentation data (seg.nrrd file)
- PNG files of each axial slice of the CT scan
- PNG files of the segmentation mask for each axial slice
- PNG files of the largest contour mask for each axial slice

## UI Specification
When the application is launched it should automatically discover every subfolder in the data folder. The user should be able to easily switch between CT scans. When a CT scan is selected it will display the data associated with the scan in 3 panels.

The first panel should show the axial slices of the CT volume from the nrrd file. There should be a slider that controls which slice is displayed. In addition to this the corresponding mask for each slice (from seg.nrrd file) should be displayed in red with 50% opacity. Their should be a button which shows and hides the mask.

The second panel should show the axial slices of the CT volume from the processed PNG files. There should be a slider that controls which slice is displayed. In addition to this the corresponding mask for each slice (from PNG mask files) should be displayed in red with 50% opacity. Their should be a button which shows and hides the mask.

The third panel should show the largest contour mask that corresponds to the slice displayed in the second panel.