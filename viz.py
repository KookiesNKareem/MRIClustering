import vtk
from vtk.util import numpy_support

def visualize_with_vtk(image_3d, spacing):
    # Convert the numpy array to a VTK array (Ensure data type is correct)
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=image_3d.ravel(), deep=True, array_type=vtk.VTK_SHORT)

    # Create a VTK image data object (Check if dimensions and orientation are correct)
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(image_3d.shape)
    vtk_image.GetPointData().SetScalars(vtk_data_array)

    vtk_image.SetSpacing(spacing[2], spacing[1], spacing[0])

    # Create a mapper
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(vtk_image)

    # Create a color transfer function (Adjust based on your data range)
    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(-1000, 0.0, 0.0, 0.0)  # Air
    color_func.AddRGBPoint(0, 0.0, 0.0, 1.0)      # Soft Tissue
    color_func.AddRGBPoint(600, 1.0, 1.0, 1.0)    # Bone

    # Create an opacity transfer function (Adjust this as well)
    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(-1000, 0.0)
    opacity_func.AddPoint(0, 0.3)
    opacity_func.AddPoint(600, 0.8)

    # The property describes how the data will look
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_func)
    volume_property.SetScalarOpacity(opacity_func)
    volume_property.SetInterpolationTypeToLinear()
    volume_property.ShadeOn()

    # The volume holds the mapper and the property and can be used to position/orient the data
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)

    # Renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the volume to the renderer
    renderer.AddVolume(volume)
    renderer.SetBackground(0, 0, 0)

    # Start the rendering
    render_window.Render()
    render_window_interactor.Start()

