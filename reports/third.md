# Report on 2D Temperature Propagation Visualization


## Introduction
The objective of this part of the project was to visualize the propagation of temperature across a two-dimensional medium over time. The dataset provided represents temperature values at discrete points on a 2D grid, capturing the dynamics of heat distribution. Visualizing this data is crucial for understanding thermal behavior, validating computational models, and identifying patterns in heat transfer processes.

First step was to create the VTK file from the serial implementation of our program, the I followed these steps.


1. **Data Reading**: It reads a VTK data file named `serial.vtk`, which contains structured points data representing a 2D grid (100x100 points) with scalar values assigned to each point.

2. **Data Mapping**: It uses `vtkDataSetMapper` to map the data from the VTK file into a format suitable for rendering. The mapper converts the scalar values into graphical primitives.

3. **Actor Creation**: An actor (`vtkActor`) is created and associated with the mapper. The actor represents an entity in the rendering scene that can be manipulated and displayed.

4. **Rendering Setup**: A renderer (`vtkRenderer`) is initialized, and the actor is added to it. The background color of the renderer is set to white.

5. **Render Window and Interactor**: A render window (`vtkRenderWindow`) is created to display the rendering, and an interactor (`vtkRenderWindowInteractor`) is set up to handle user interactions (although in this case, the interactor isn't actively used).

6. **Rendering and Image Capture**: The scene is rendered using `render_window.Render()`. Then, `vtkWindowToImageFilter` captures the content of the render window as an image.

7. **Image Saving**: The captured image is saved as a PNG file named `serial.png` using `vtkPNGWriter`.

**About the Data (`serial.vtk`):**

- The VTK file defines a structured points dataset with dimensions 100x100x1, essentially creating a 2D grid.
- The origin is at (0, 0, 0), and the spacing between points is uniform (1 unit in each direction).
- It contains scalar data named `u`, which is a floating-point value assigned to each of the 10,000 points in the grid.
- The scalar values are all under the max bound of `100.0`, indicating a field across the entire grid.

**What the Program Does Overall:**

The program reads a scalar field defined over a 2D grid from a VTK file and visualizes it. The program then captures this visualization as an image and saves it to a file named `serial.png`.


![](../output.png)


**Summary:**

- **Input**: A VTK file (`serial.vtk`) containing a 2D grid with varied scalar values.
- **Process**: Reads the data, maps it for rendering, renders the scene, captures the render window as an image.
- **Output**: An image file (`serial.png`) displaying the visualization of the scalar field.

This program effectively converts a structured scalar dataset into a visual representation and saves it as an image.
