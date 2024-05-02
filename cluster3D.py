import vtk
import numpy as np
import skfuzzy as fuzz
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

color_map = {
    "T1": {
        # WRONG FIX!!
        "CSF": [1.0, 0.0, 0.0],  # Red
        "Gray Matter": [0.0, 1.0, 0.0],  # Green
        "Bone": [0.0, 0.0, 1.0],          # Blue
        "White Matter": [0.5, 0.5, 0.0]           # Yellow
         },
    "T2": {
        "Gray Matter": [1.0, 0.0, 0.0],  # Red
        "White Matter": [0.0, 1.0, 0.0],  # Green
        "CSF": [0.0, 0.0, 1.0],          # Blue
        "Bone": [0.5, 0.5, 0.0]           # Yellow
    }
    # Add more MRI types if necessary
}

def numpy_to_vtk_image_data(numpy_data):
    """Convert a numpy array to vtkImageData."""
    # Create a new vtkImageData object
    vtk_data = vtk.vtkImageData()

    # Set the dimensions of the vtkImageData object
    depth, height, width = numpy_data.shape
    vtk_data.SetDimensions(width, height, depth)

    # Allocate the necessary space for the data
    vtk_data.AllocateScalars(vtk.VTK_FLOAT, 1)

    # Copy data from numpy array to vtkImageData
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                scalar_value = numpy_data[z, y, x]
                vtk_data.SetScalarComponentFromFloat(x, y, z, 0, scalar_value)

    return vtk_data


def load_dicom_series(directory):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(directory)
    reader.Update()
    return reader.GetOutput()


def create_text_labels(renderer, color_map, mri_type, window_width, window_height):
    spacing = 20  # Space between labels
    y_position = window_height - 40  # Starting y position

    for tissue, color in color_map[mri_type].items():
        # Create a text actor
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(tissue)
        text_actor.GetTextProperty().SetFontSize(20)
        text_actor.GetTextProperty().SetColor(color)
        text_actor.SetPosition(spacing, y_position)

        # Add the text actor to the renderer
        renderer.AddActor(text_actor)

        # Update y_position for the next label
        y_position -= 30  # Adjust the spacing between labels as needed


def create_3d_model(image_data, weight):
    # Create a mapper
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(image_data)

    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)  # Assuming 0 is the background
    colorFunc.AddRGBPoint(1, 1.0, 0.0, 0.0)  # First cluster
    colorFunc.AddRGBPoint(2, 0.0, 1.0, 0.0)  # Second cluster, and so on...
    colorFunc.AddRGBPoint(3, 0.0, 0.0, 1.0)  # Third cluster, and so on...
    colorFunc.AddRGBPoint(4, 0.5, 0.5, 0.0)  # Third cluster, and so on...

    opacityFunc = vtk.vtkPiecewiseFunction()
    opacityFunc.AddPoint(0, 0.0)  # Fully transparent for background
    opacityFunc.AddPoint(1, 0.75)  # Semi-transparent for first cluster
    opacityFunc.AddPoint(2, 0.75)  # Adjust these values as needed
    opacityFunc.AddPoint(3, 0.75)  # Semi-transparent for first cluster
    opacityFunc.AddPoint(4, 0.75)  # Adjust these values as needed

    # The property describes how the data will look
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(opacityFunc)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    # The volume holds the mapper and the property and can be used to position/orient the volume
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volumeProperty)

    # Renderer and render window
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)

    # Create and add text labels
    create_text_labels(renderer, color_map, weight, 800, 800)

    # Render window interactor
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)

    # Interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    renderInteractor.SetInteractorStyle(style)

    # Add UI interactor
    uiInteractor = vtk.vtkVolumePicker()
    uiInteractor.SetTolerance(0.005)
    renderInteractor.SetPicker(uiInteractor)

    renderer.AddVolume(volume)
    renderer.SetBackground(0, 0, 0)
    renderWin.SetSize(800, 800)

    # Initialize and start the interaction
    renderInteractor.Initialize()
    renderWin.Render()
    renderInteractor.Start()


def vtk_image_data_to_numpy(vtk_data):
    """Convert vtkImageData to a numpy array."""
    dims = vtk_data.GetDimensions()
    num_elements = dims[0] * dims[1] * dims[2]
    numpy_array = np.zeros(num_elements, dtype=float)

    # Copy data from vtkData to numpy array
    for i in range(num_elements):
        numpy_array[i] = vtk_data.GetScalarComponentAsDouble(i % dims[0], (i // dims[0]) % dims[1],
                                                             i // (dims[0] * dims[1]), 0)

    # Reshape the numpy array to 3D using dimensions of vtk_data
    numpy_array = numpy_array.reshape(dims[2], dims[1], dims[0])

    return numpy_array


def apply_fuzzy_cmeans_to_slice(slice_data, n_clusters=4, max_iter=1000, error=0.005, m=2.0):
    """Apply fuzzy c-means clustering to a 2D slice."""
    # Normalize and reshape data for clustering
    slice_data_normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
    flat_data = slice_data_normalized.flatten().reshape(-1, 1)

    # Apply Fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(flat_data.T, n_clusters, m, error, max_iter, init=None)

    # Assign pixels to clusters
    cluster_membership = np.argmax(u, axis=0)
    clustered_slice = cluster_membership.reshape(slice_data.shape)

    return clustered_slice


def cluster_entire_volume(volume_data):
    """Apply clustering to each slice and combine into a 3D volume."""
    clustered_slices = [apply_fuzzy_cmeans_to_slice(slice_data) for slice_data in volume_data]
    return np.stack(clustered_slices)


Tk().withdraw()
dicom_dir = askdirectory(initialdir="/Users/kfareed24/PycharmProjects/clustering/MRIs")
if dicom_dir == "":
    raise FileNotFoundError

num = dicom_dir[dicom_dir.rfind("/") + 1:]
if os.path.exists(f'clustered/clustered_data{num}.npy'):
    clustered_data = np.load(f'clustered/clustered_data{num}.npy')
else:
    vtk_data = load_dicom_series(dicom_dir)
    numpy_data = vtk_image_data_to_numpy(vtk_data)
    clustered_data = cluster_entire_volume(numpy_data)
    np.save(f'clustered/clustered_data{num}.npy', clustered_data)

# Create 3D model with clustered data
vtk_clustered_data = numpy_to_vtk_image_data(clustered_data)  # Function to convert numpy array to vtkImageData
create_3d_model(vtk_clustered_data, "T2")
