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

from arena_api import enums
from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.enums import PixelFormat
from common.lucid_vision import (
    TAB1,TAB2,TRITON,HELIOS2,
    create_devices_with_tries,
    get_applicable_devices,
    convert_buffer_to_Coord3D_ABCY16,
    convert_buffer_to_Mono8,
    read_calibration,
    write_orientation,
)
from common.node_overrides import CameraOverrides
TIMEOUT = 2000

FILE_NAME_IN = 'tritoncalibration.yml'

FILE_NAME_OUT = "orientation.yml"

def get_image_HLT(device):

    # Configure stream and pixel format with auto-restore --------------------
    nodemap = device.nodemap
    with CameraOverrides(
        device,
        nodes={'PixelFormat': PixelFormat.Coord3D_ABCY16},
        stream_nodes={
            'StreamAutoNegotiatePacketSize': True,
            'StreamPacketResendEnable': True,
        },
        strict=False,
    ):

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


        device.start_stream()
        buffer = device.get_buffer()

    buffer_Coord3D_ABCY16 = convert_buffer_to_Coord3D_ABCY16(buffer)

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

    return intensity_image, xyz_mm


def get_image_TRI(device):
    '''
    Returns image from a Triton device
    '''

    # Configure stream and pixel format with auto-restore --------------------
    nodemap = device.nodemap
    with CameraOverrides(
        device,
        nodes={'PixelFormat': PixelFormat.RGB8},
        stream_nodes={
            'StreamAutoNegotiatePacketSize': True,
            'StreamPacketResendEnable': True,
        },
        strict=False,
    ):

        # Get image ---------------------------------------------------
        device.start_stream()
        buffer = device.get_buffer()
        buffer_Mono8 = convert_buffer_to_Mono8(buffer)
        buffer_bytes_per_pixel = int(len(buffer_Mono8.data)/(buffer_Mono8.width * buffer_Mono8.height))
        image_matrix = np.asarray(buffer_Mono8.data, dtype=np.uint8)
        image_matrix_reshaped = image_matrix.reshape(buffer_Mono8.height, buffer_Mono8.width, buffer_bytes_per_pixel)

        # Stop stream -------------------------------------------------
        device.requeue_buffer(buffer)
        device.stop_stream()

    return image_matrix_reshaped

def find_calibration_points_HLT(image_in_orig):
    '''
    Returns an array of calibration points found in the given image captured by Helios2
    '''

    image_in = image_in_orig

    # Create blob detector ------------------------------------------------------------------
    bright_params = cv2.SimpleBlobDetector_Params()
    bright_params.filterByColor = True
    bright_params.blobColor = 255 # White circles in the calibration target
    bright_params.thresholdStep = 2
    bright_params.minArea = 10.0 # Min/max area can be adjusted based on size of dots in image
    bright_params.maxArea = 1000.0

    blob_detector = cv2.SimpleBlobDetector.create(bright_params)

    # pattern_size(num_cols, num_rows) num_cols: number of columns (number of
    # circles in a row) of the calibration target viewed by the camera num_rows:
    # number of rows (number of circles in a column) of the calibration target
    # viewed by the camera Specify according to the orientation of the
    # calibration target
    pattern_size = (5, 4)

    # Find min and max values in the input image
    _, max_value, _, _ = cv2.minMaxLoc(image_in)
    
    # Scale image to 8-bit, using full 8-bit range
    image_8bit = cv2.convertScaleAbs(image_in, alpha=255.0/max_value)

    is_found, grid_centers = cv2.findCirclesGrid(image_8bit, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=blob_detector)

    return is_found, grid_centers


def find_calibration_points_TRI(image_in_orig):
    '''
    Returns an array of calibration points found in the given image captured by Triton
    '''

    scaling = 1.0
    image_in = image_in_orig
    num_cols_orig = image_in_orig.shape[1] # width
    num_rows_orig = image_in_orig.shape[0] # height

    # Create blob detector ------------------------------------------------------------------
    bright_params = cv2.SimpleBlobDetector_Params()
    bright_params.filterByColor = True
    bright_params.blobColor = 255 # White circles in the calibration target
    bright_params.filterByCircularity = True
    bright_params.minCircularity = 0.8

    blob_detector = cv2.SimpleBlobDetector.create(bright_params)

    # Find calibration points --------------------------------------------------------
    pattern_size = (5, 4) # (pattern_per_row, pattern_per_column)
    is_found, grid_centers = cv2.findCirclesGrid(image_in, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=blob_detector)

    scaled_nrows = 2400.0

    while not is_found and scaled_nrows >= 100:
        scaled_nrows /= 2.0
        scaling = float(num_rows_orig / scaled_nrows)

        image_in = cv2.resize(image_in_orig, (int(num_cols_orig/scaling), int(num_rows_orig/scaling))) # cv2.resize(image, (width, height))

        is_found, grid_centers = cv2.findCirclesGrid(image_in, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=blob_detector)

    if is_found:
        for center in grid_centers:
            center[0][0] *= scaling
            center[0][1] *= scaling

    return is_found, grid_centers
    

'''
EXAMPLE
'''
def calculate_and_save_orientation_values(device_triton, device_helios2):
    
    # Read in camera matrix and distance coefficient ----------------------------------------------------------------
    print(f'{TAB1}Read camera matrix and distance coefficients from file {FILE_NAME_IN}')
    camera_matrix, dist_coeffs = read_calibration(FILE_NAME_IN)

    # Get an image from Helios2 ----------------------------------------------------------------
    print(f'{TAB1}Get and prepare HLT image')
    image_matrix_HLT_intensity, image_matrix_HLT_XYZ = get_image_HLT(device_helios2)

    # Get an image from Triton ----------------------------------------------------------------
    print(f'{TAB1}Get and prepare TRI image')
    image_matrix_TRI = get_image_TRI(device_triton)

    # Calculate orientation values ----------------------------------------------------------------
    print(f'{TAB1}Calculate orientation values')

    # Find HLT calibration points using HLT intensity image
    print(f'{TAB1}Find points in HLT image')
    count = 0
    while True:
        grid_centers_HLT_found, grid_centers_HLT = find_calibration_points_HLT(image_matrix_HLT_intensity)
        if not grid_centers_HLT_found or len(grid_centers_HLT) != 20:
            print(f'{TAB2}Unable to find points in HLT intensity image. {count} seconds passed', end='\r')
            count += 1
            time.sleep(1)
        else:
            break

    # Find TRI calibration points
    print(f'{TAB1}Find points in TRI image')
    count = 0
    while True:
        grid_centers_TRI_found, grid_centers_TRI = find_calibration_points_TRI(image_matrix_TRI)
        if not grid_centers_TRI_found or len(grid_centers_TRI) != 20:
            print(f"{TAB2}Unable to find points in TRI image. {count} seconds passed", end='\r')
            count += 1
            time.sleep(1)
        else:
            break

    # Prepare for PnP
    print(f'{TAB1}Prepare for PnP')

    target_points_3d_mm = []
    target_points_3d_pixels = []
    target_points_2d_pixels = []


    for i in range(len(grid_centers_TRI)):
        c1 = round(grid_centers_HLT[i][0][0])
        r1 = round(grid_centers_HLT[i][0][1])
        c2 = round(grid_centers_TRI[i][0][0])
        r2 = round(grid_centers_TRI[i][0][1])

        [x, y, z] = [image_matrix_HLT_XYZ[r1, c1][j] for j in range(3)]
        pt = [x, y, z]
        
        print(f'{TAB2}Point {i}: {pt}')

        target_points_3d_mm.append(pt)
        target_points_3d_pixels.append(grid_centers_HLT[i][0])
        target_points_2d_pixels.append(grid_centers_TRI[i][0])

    target_points_3d_mm = np.array(target_points_3d_mm)
    target_points_3d_pixels = np.array(target_points_3d_pixels)
    target_points_2d_pixels = np.array(target_points_2d_pixels)
    orientation_succeeded, rotation_vector, translation_vector = cv2.solvePnP(target_points_3d_mm, target_points_2d_pixels, camera_matrix, dist_coeffs)

    print(f'{TAB1}Orientation succeeded' if orientation_succeeded else f'{TAB1}Orientation failed')

    # Save orientation information ----------------------------------------------------------------
    print(f'{TAB1}Save camera matrix, distance coefficients, and rotation and translation vectors to file {FILE_NAME_OUT}')
    write_orientation(FILE_NAME_OUT, camera_matrix, dist_coeffs, rotation_vector, translation_vector)



def example_entry_point():
    
    print(f'{TAB1}py_HLTRGB_2_Orientation')
    
    # Check for input file ------------------------------------------------------
    if not exists(FILE_NAME_IN):
        print(f'{TAB1}File \'{FILE_NAME_IN}\' not found. Please run example \'py_HLTRGB_1_calibration\' prior to this one.')
        return

    # Get connected devices ---------------------------------------------------
    # Create devices
    devices = create_devices_with_tries()

    # Filter applicable devices
    applicable_devices_triton = get_applicable_devices(devices, TRITON)
    applicable_devices_helios2 = get_applicable_devices(devices, HELIOS2)
    
    # Select a Triton camera to use
    device_triton = system.select_device(applicable_devices_triton)
    device_helios2 = system.select_device(applicable_devices_helios2)

    # Calculate and save orientation values ----------------------------------------
    calculate_and_save_orientation_values(device_triton, device_helios2)

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