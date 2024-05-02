import os
import pydicom
import numpy as np
import vtk
import skfuzzy as fuzz


def load_dicom_series(directory):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(directory)
    reader.Update()
    return reader.GetOutput()


def create_3d_model(image_data):
    # Create a mapper
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(image_data)

    # Create a color transfer function
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(500, 1.0, 0.5, 0.3)
    colorFunc.AddRGBPoint(1000, 1.0, 0.5, 0.3)
    colorFunc.AddRGBPoint(1150, 1.0, 1.0, 0.9)

    # Create an opacity transfer function
    opacityFunc = vtk.vtkPiecewiseFunction()
    opacityFunc.AddPoint(0, 0.00)
    opacityFunc.AddPoint(500, 0.15)
    opacityFunc.AddPoint(1000, 0.15)
    opacityFunc.AddPoint(1150, 0.85)

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


# Directory containing your DICOM files
dicom_dir = "MRIs/3215/Ax_T2_FSE/2011-08-18_13_40_02.0/I373992"

# Load the DICOM series
image_data = load_dicom_series(dicom_dir)

# Create the 3D model
create_3d_model(image_data)
