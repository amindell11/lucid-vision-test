# -----------------------------------------------------------------------------
# Copyright (c) 2024, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

import time
import sys
import ctypes
from os.path import exists

import numpy as np # pip3 install numpy
import cv2  # pip3 install opencv-python
from matplotlib import pyplot as plt # pip3 install matplotlib
# pip3 install pillow
from PIL import Image as PIL_Image
from PIL import ImageTk as PIL_ImageTk
# pip3 install tk / or 'sudo apt-get install python3-tk' for linux
from tkinter import *
from enum import Enum
import math

from arena_api import enums
from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.enums import PixelFormat
from arena_api.__future__.save import Writer

'''
Helios RGB: Overlay
    This example is part 3 of a 3-part example on color overlay over 3D images.
    With the system calibrated, we can now remove the calibration target from
    the scene and grab new images with the Helios and Triton cameras, using the
    calibration result to find the RGB color for each 3D point measured with
    the Helios. Based on the output of solvePnP we can project the 3D points
    measured by the Helios onto the RGB camera image using the OpenCV function
    projectPoints. Grab a Helios image with the GetHeliosImage()
    function(output: xyz_mm) and a Triton RGB image with the
    GetTritionRGBImage() function(output: triton_rgb). The following code shows
    how to project the Helios xyz points onto the Triton image, giving a(row,
    col) position for each 3D point. We can sample the Triton image at
    that(row, col) position to find the 3D point’s RGB value.
'''

'''
Settings
'''
TAB1 = "  "
TAB2 = "    "

# image timeout
TIMEOUT = 2000

# calibration values file name
FILE_NAME_IN = "orientation.yml"

# orientation values file name
FILE_NAME_OUT = "py_HLTRGB_3_Overlay.ply"

TRITON = 'Triton'
HELIOS2 = 'Helios2'

'''
PREPARATION
'''
def create_devices_with_tries():
    '''
    This function waits for the user to connect a device before raising
        an exception
    '''

    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    while tries < tries_max:  # Wait for device for 60 seconds
        devices = system.create_device()
        if not devices:
            print(
                f'{TAB1}Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                f'secs for a device to be connected!')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                print(f'{TAB1}{sec_count + 1 } seconds passed ',
                    '.' * sec_count, end='\r')
            tries += 1
        else:
            print(f'{TAB1}Created {len(devices)} device(s)')
            return devices
    else:
        raise Exception(f'{TAB1}No device found! Please connect a device and run '
                        f'the example again.')


def is_applicable_device_triton(device):
    '''
    Return True if a device is a Triton camera, False otherwise
    '''
    model_name = device.nodemap.get_node('DeviceModelName').value
    return "TRI" in model_name and "-C" in model_name


def is_applicable_device_helios2(device):
    '''
    Return True if a device is a Helios2 camera, False otherwise
    '''
    model_name = device.nodemap.get_node('DeviceModelName').value
    return "HLT" in model_name or "HT" in model_name 


def get_applicable_devices(devices, type):
    '''
    Return a list of applicable Triton devices
    '''
    applicable_devices = []

    for device in devices:
        if type == TRITON and is_applicable_device_triton(device):
            applicable_devices.append(device)
        elif type == HELIOS2 and is_applicable_device_helios2(device):
            applicable_devices.append(device)
    
    if not len(applicable_devices):
        raise Exception(f'{TAB1}No applicable device found! Please connect an Triton and Helios2 device and run '
                        f'the example again.')

    print(f'{TAB1}Detected {len(applicable_devices)} applicable {type} device(s)')
    return applicable_devices


'''
HELPERS
'''
def convert_buffer_to_Coord3D_ABCY16(buffer):
    '''
    Convert to Coord3D_ABCY16 format
    '''
    if buffer.pixel_format == enums.PixelFormat.Coord3D_ABCY16:
        return buffer
    print(f'{TAB1}Converting image buffer pixel format to Coord3D_ABCY16')
    return BufferFactory.convert(buffer, enums.PixelFormat.Coord3D_ABCY16)


def get_image_HLT(device):
    '''
    Returns intensity and xyz images from a Helios2 device
    '''
    # Set device stream nodemap --------------------------------------------
    tl_stream_nodemap = device.tl_stream_nodemap
    # Enable stream auto negotiate packet size
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    # Enable stream packet resend
    tl_stream_nodemap['StreamPacketResendEnable'].value = True

    # Set nodes --------------------------------------------------------------
    # - pixelformat to Coord3D_ABCY16
    # - 3D operating mode
    nodemap = device.nodemap
    nodemap.get_node('PixelFormat').value = PixelFormat.Coord3D_ABCY16

    # Get node values ---------------------------------------------------------
    # Read the scale factor and offsets to convert from unsigned 16-bit values 
    # in the Coord3D_ABCY16 pixel format to coordinates in mm

        # "Coord3D_ABCY16s" and "Coord3D_ABCY16" pixelformats have 4
        # channels per pixel. Each channel is 16 bits and they represent:
        #   - x position
        #   - y postion
        #   - z postion
        #   - intensity

    # get the coordinate scale in order to convert x, y and z values to millimeters as
    # well as the offset for x and y to correctly adjust values when in an
    # unsigned pixel format
    print(f'{TAB1}Get xyz coordinate scales and offsets from nodemap')
    xyz_scale_mm = nodemap["Scan3dCoordinateScale"].value # Coordinate scale to convert x, y, and z values to mm
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateA"
    x_offset_mm = nodemap["Scan3dCoordinateOffset"].value # offset for x to adjust values when in unsigned pixel format
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateB"
    y_offset_mm = nodemap["Scan3dCoordinateOffset"].value # offset for y
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
    z_offset_mm = nodemap["Scan3dCoordinateOffset"].value # offset for z


    # Start stream and get image
    device.start_stream()
    buffer = device.get_buffer()
    buffer_copy = BufferFactory.copy(buffer)

    # Copy image buffer into the Coord3d_ABCY16 format
    buffer_Coord3D_ABCY16 = convert_buffer_to_Coord3D_ABCY16(buffer)

    # get height and width
    height = int(buffer_Coord3D_ABCY16.height)
    width = int(buffer_Coord3D_ABCY16.width)
    channels_per_pixel = int(buffer_Coord3D_ABCY16.bits_per_pixel / 16)

    xyz_mm = np.zeros((height, width, 3), dtype=np.float32)
    intensity_image = np.zeros((height, width), dtype=np.uint16)

    # get input data
    # Buffer.pdata is a (uint8, ctypes.c_ubyte) pointer.
    # This pixelformat has 4 channels, and each channel is 16 bits.
    # It is easier to deal with Buffer.pdata if it is cast to 16bits
    # so each channel value is read correctly.
    # The pixelformat is suffixed with "S" to indicate that the data
    # should be interpereted as signed. This one does not have "S", so
    # we cast it to unsigned.
    pdata_as_uint16 = ctypes.cast(buffer_Coord3D_ABCY16.pdata, ctypes.POINTER(ctypes.c_uint16))

    i = 0

    for ir in range(height):
        for ic in range(width):

            # Get unsigned 16 bit values for X,Y,Z coordinates
            x_u16 = pdata_as_uint16[i]
            y_u16 = pdata_as_uint16[i + 1]
            z_u16 = pdata_as_uint16[i + 2]

            # Convert 16-bit X,Y,Z to float values in mm
            xyz_mm[ir, ic][0] = float(x_u16 * xyz_scale_mm + x_offset_mm)
            xyz_mm[ir, ic][1] = float(y_u16 * xyz_scale_mm + y_offset_mm)
            xyz_mm[ir, ic][2] = float(z_u16 * xyz_scale_mm + z_offset_mm)

            intensity_image[ir, ic] = pdata_as_uint16[i + 3]

            i += channels_per_pixel


    # Stop stream
    device.requeue_buffer(buffer)
    device.stop_stream()

    return intensity_image, xyz_mm, height, width, buffer_copy


def convert_buffer_to_Mono8(buffer):
    '''
    Convert to Mono8 format
    '''
    if (buffer.pixel_format == enums.PixelFormat.Mono8):
        return buffer
    print(f'{TAB1}Converting image buffer pixel format to Mono8 ')
    return BufferFactory.convert(buffer, enums.PixelFormat.Mono8)


def get_image_TRI(device):
    '''
    Returns image from a Triton device
    '''
    # Set nodes --------------------------------------------------------------
    # - pixelformat to RGB8
    # - 3D operating mode
    nodemap = device.nodemap
    nodemap.get_node('PixelFormat').value = PixelFormat.RGB8

    # Set device stream nodemap --------------------------------------------
    tl_stream_nodemap = device.tl_stream_nodemap
    # Enable stream auto negotiate packet size
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    # Enable stream packet resend
    tl_stream_nodemap['StreamPacketResendEnable'].value = True

    # Get image ---------------------------------------------------
    device.start_stream()
    buffer = device.get_buffer()
    buffer_bytes_per_pixel = int(len(buffer.data)/(buffer.width * buffer.height))
    image_matrix = np.asarray(buffer.data, dtype=np.uint8)
    image_matrix_reshaped = image_matrix.reshape(buffer.height, buffer.width, buffer_bytes_per_pixel)

    # Stop stream -------------------------------------------------
    device.requeue_buffer(buffer)
    device.stop_stream()

    return image_matrix_reshaped


'''
EXAMPLE
'''
def overlay_color_onto_3D_and_save(device_triton, device_helios2):
    
    # Get node values that will be changed in order to return their values at the end of the example ----------------------------------------------------------------
    nodemap_triton = device_triton.nodemap
    nodemap_helios2 = device_helios2.nodemap
    pixel_format_initial_triton = nodemap_triton.get_node("PixelFormat").value
    pixel_format_initial_helios2 = nodemap_helios2.get_node("PixelFormat").value
    
    # Read in camera matrix, distance coefficients, rotation and translation vectors ----------------------------------------------------------------
    print(f'{TAB1}Read camera matrix, distance coefficients, rotation and translation vectors from file {FILE_NAME_IN}')
    fs = cv2.FileStorage(FILE_NAME_IN, cv2.FileStorage_READ)
    camera_matrix = fs.getNode('cameraMatrix').mat()
    dist_coeffs = fs.getNode('distCoeffs').mat()
    rotation_vector = fs.getNode('rotationVector').mat()
    translation_vector = fs.getNode('translationVector').mat()
    fs.release()

    # Get an image from Helios2 ----------------------------------------------------------------
    print(f'{TAB1}Get and prepare HLT image')
    _, image_matrix_XYZ, height, width, p_image_HLT = get_image_HLT(device_helios2)
    cv2.imwrite(FILE_NAME_OUT.strip('.ply')+'_XYZ.jpg', image_matrix_XYZ)

    # Get an image from Triton ----------------------------------------------------------------
    print(f'{TAB1}Get and prepare TRI image')
    image_matrix_RGB = get_image_TRI(device_triton)
    cv2.imwrite(FILE_NAME_OUT.strip('.ply')+'_RGB.jpg', image_matrix_RGB)

    # Overlay RGB color data onto 3D XYZ points ----------------------------------------------------------------
    print(f'{TAB1}Overlay RGB color data onto 3D XYZ points')

    # Reshape image matrix
    # Convert the Helios xyz values from 640x480 to a Nx1 matrix to feed into projectPoints
    print(f'{TAB2}Reshape XYZ matrix')
    size = image_matrix_XYZ.shape[0] * image_matrix_XYZ.shape[1]
    xyz_points = np.reshape(image_matrix_XYZ, (size, 3))

    # Project points
    # Use projectPoints to find the position in the Triton image (row,col) of each Helios 3d point
    print(f'{TAB2}Project points')
    project_points_TRI, _ = cv2.projectPoints(xyz_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Finally, loop through the set of points and access the Triton RGB image at the positions
	# calculated by projectPoints to find the RGB value of each 3D point
    print(f'{TAB2}Get values at projected points')
    CustomArrayType = (ctypes.c_byte * (height * width * 3))
    color_data = CustomArrayType()

    for i in range(height * width):
        col_TRI = round(project_points_TRI[i][0][0])
        row_TRI = round(project_points_TRI[i][0][1])

        # Only handle appropriate points
        if row_TRI < 0 or col_TRI < 0 or row_TRI >= image_matrix_RGB.shape[0] or col_TRI >= image_matrix_RGB.shape[1]:
            continue

        # Access corresponding XYZ and RGB data
        r_val = image_matrix_RGB[row_TRI, col_TRI][0]
        g_val = image_matrix_RGB[row_TRI, col_TRI][1]
        b_val = image_matrix_RGB[row_TRI, col_TRI][2]

        # Now you have the RGB values of a measured 3D Point at location (X,Y,Z).
		# Depending on your application you can do different things with these values,
		# for example, feed them into a point cloud rendering engine to view a 3D RGB image.

        # Grab RGB data to save colored .ply
        color_data[i*3] = r_val
        color_data[i*3+1] = g_val
        color_data[i*3+2] = b_val
        

    # Save result ----------------------------------------------------------------
    print(f'{TAB1}Save image to {FILE_NAME_OUT}')

    # Prepare to save
    # create an image writer
    # When saving as .ply file, the writer optionally can take width, 
    # height, and bits per pixel of the image(s) it would save. 
    # if these arguments are not passed at run time, the first buffer passed 
    # to the Writer.save() function will configure the writer to the arguments 
    # buffer's width, height, and bits per pixel
    writer_ply = Writer()

    # uint8_ptr = ctypes.POINTER(ctypes.c_ubyte)
    # p_color_data = uint8_ptr(color_data)
    # Create p_color_data array
    p_color_data = (ctypes.c_ubyte * len(color_data)).from_address(ctypes.addressof(color_data))

    # Save .ply with color data
    # save.py > Writer > save
    # xwriter.py > Save > _SaveWithColor
    # const uint8_t* pColor
    # Also example in py_helios_heatmap.py
    # and py_save_writer_ply.py
    writer_ply.save(p_image_HLT, FILE_NAME_OUT, color=p_color_data, filter_points=True)

    # Return nodes to their original values ----------------------------------------------------------------
    nodemap_triton.get_node("PixelFormat").value = pixel_format_initial_triton
    nodemap_helios2.get_node("PixelFormat").value = pixel_format_initial_helios2


def example_entry_point():
    
    print(f'{TAB1}py_HLTRGB_3_Overlay')
    
    # Check for input file ------------------------------------------------------
    if not exists(FILE_NAME_IN):
        print(f'{TAB1}File \'{FILE_NAME_IN}\' not found. Please run example \'py_HLTRGB_1_calibration\' and \'py_HLTRGB_2_orientation\' prior to this one.')
        return

    # Get connected devices ---------------------------------------------------
    # Create a device
    devices = create_devices_with_tries()

    # Filter applicable devices
    applicable_devices_triton = get_applicable_devices(devices, TRITON)
    applicable_devices_helios2 = get_applicable_devices(devices, HELIOS2)
    
    # Select a Triton camera to use
    device_triton = system.select_device(applicable_devices_triton)
    device_helios2 = system.select_device(applicable_devices_helios2)

    # Overlay color onto 3D and save ----------------------------------------
    overlay_color_onto_3D_and_save(device_triton, device_helios2)

    # Clean up ----------------------------------------------------------------
    '''
    Destroy device. This call is optional and will automatically be
        called for any remaining devices when the system module is unloading.
    '''
    system.destroy_device()
    print(f'{TAB1}Destroyed all created devices')
    

if __name__ == '__main__':
    print('Example started\n')
    example_entry_point()
    print('\nExample finished successfully')