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

'''
Helios RGB: TritonCalibration
    This example is part 1 of a 3-part example on color overlay over 3D images.
    Before the data between the two cameras can be combined,
    we must first calibrate the lens on the Triton color camera to find its
    optical center and focal length (intrinsics), and lens distortion
    coefficients (pinhole model). We can achieve this by printing a target with
    a checkerboard pattern or you can download our calibration target here
    (15kB, PDF, 8.5 x 11 in)
    https:arenasdk.s3-us-west-2.amazonaws.com/LUCID_target_whiteCircles.pdf
    Before calibrating the Triton camera you must focus its lens. Place the
    target at your application's working distance and focus the Triton's
    lens so that the calibration target is in focus. Calibrating the Triton
    camera requires grabbing several images of the calibration chart at
    different positions within the camera's field of view. At least 3 images
    are required but 4 to 8 images are typically used to get a better - quality
    calibration.
'''
TAB1 = "  "
TAB2 = "    "

# image timeout
TIMEOUT = 2000

# number of calibration points to compare
NUM_IMAGES = 10

# calibration values file name
FILE_NAME = 'tritoncalibration.yml'

# time to sleep between images (in seconds)
SLEEP_SECOND = 1

'''
HELPERS
'''
# Helper class
class Settings:
    class Pattern(Enum):
        NOT_EXISTING = 1
        CHESSBOARD = 2
        CIRCLES_GRID = 3
        ASYMMETRIC_CIRCLES_GRID = 4
    
    class InputType(Enum):
        INVALID = 1
        CAMERA = 2
        VIDEO_FILE = 3
        IMAGE_LIST = 4

    def __init__(self):
        self.good_input = False

        self.board_size = {'width': 0, 'height': 0} # The size of the board -> number of items by width and height
        self.calibration_pattern = Settings.Pattern.NOT_EXISTING # One of the Chessboard, circles, or asymmetric circle pattern
        self.square_size = None # The size of a square in your defined unit (point, millimeter, etc).
        self.nr_frames = None # The number of frames to use from the input for calibration
        self.aspect_ratio = None # The aspect ratio
        self.delay = 0 # In case of a video input
        self.writePoints = False # Write detected feature points
        self.calib_zero_target_dist = False # Assume zero tangential distortion
        self.calib_fix_principal_point = False # Fix the principal piont at the center
        self.flip_vertical = False # Flip the captured images around the horizontal axis
        self.output_filename = None
        self.show_undistorsed = None # Show undistorted images after calibration
        self.input = None # The input ->

        self.use_fisheye = False # Use fisheye camera model for calibration
        self.fix_k1 = False # Fix K1 distortion coefficient
        self.fix_k2 = False # Fix K2 distortion coefficient
        self.fix_k3 = False # Fix K3 distortion coefficient
        self.fix_k4 = False # Fix K4 distortion coefficient
        self.fix_k5 = False # Fix K5 distortion coefficient

        self.camera_ID = None
        self.image_list = []
        self.at_image_list = None
        self.input_capture = None
        self.input_type = None
        self.good_input = False
        self.flag = 0

        self.pattern_to_use = None
    

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


def get_applicable_devices(devices):
    '''
    Return a list of applicable Triton devices
    '''
    applicable_devices = []

    for device in devices:
        if is_applicable_device(device):
            applicable_devices.append(device)
    
    if not len(applicable_devices):
        raise Exception(f'{TAB1}No applicable device found! Please connect an Triton device and run '
                        f'the example again.')

    print(f'{TAB1}Detected {len(applicable_devices)} applicable device(s)')
    return applicable_devices


def is_applicable_device(device):
    '''
    Return True if a device is a Triton camera, False otherwise
    '''
    model_name = device.nodemap.get_node('DeviceModelName').value
    return 'TRI' in model_name and '-C' in model_name


def convert_buffer_to_Mono8(buffer):
    '''
    Convert bufferto Mono8 format
    '''
    if buffer.pixel_format == enums.PixelFormat.Mono8:
        return buffer
    print(f'{TAB1}Converting image buffer pixel format to Mono8 ')
    return BufferFactory.convert(buffer, enums.PixelFormat.Mono8)


def find_calibration_points(image_in_orig):
    '''
    Returns an array of calibration points found in the given image
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


def compute_reprojection_err(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs, fisheye):
    '''
    Compute reprojection error and its total average value
    '''
    
    per_view_errors = [0] * len(object_points)
    image_points2 = np.array([])
    total_points = 0
    total_error = 0

    for i in range(len(object_points)):
        if fisheye:
            image_points2, _ = cv2.fisheye.projectPoints(object_points[i], image_points2, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        else:
            image_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

        error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2)

        n = len(object_points[i])
        per_view_errors[i] = math.sqrt(error * error / n)
        total_error += error * error
        total_points += n
    
    return math.sqrt(total_error / total_points), per_view_errors


def calc_board_corner_positions(board_size, square_size):
    '''
    Returns an array that indicates the position of the grids
    '''
    corners = []

    for i in range(board_size['height']):
        for j in range(board_size['width']):
            corners.append(np.array((j*square_size, i*square_size, 0), dtype=np.float32))

    return np.array(corners)


def calculate(s: Settings, image_size, image_points):
    '''
    Calculate camera matrix, dist coefficients and total average error
    '''
    
    # ! [fixed_aspect]
    camera_matrix = np.eye(3, dtype=np.float64)
    if s.flag & cv2.CALIB_FIX_ASPECT_RATIO:
        camera_matrix[0, 0] = s.aspect_ratio

    # ! [fixed_aspect]
    np_shape = (4, 1) if s.use_fisheye else (8, 1)
    dist_coeffs = np.zeros(np_shape, dtype=np.float64)

    # Specify the size of the calibration board and distance between grid circles
    s.board_size['width'] = 5
    s.board_size['height'] = 4
    s.square_size = 50 # distance between grids in mm

    # Find the grid point positions and make the size of the object_points array the same as the image_points array
    object_point = calc_board_corner_positions(s.board_size, s.square_size)
    object_points = [object_point] * len(image_points)

    # Ensure the size of elements in object_points and image_points matches
    for i in range(len(object_points)):
        if len(object_points[i]) != len(image_points[i]):
            raise ValueError('object point and image point do not share shape')

    # Find intrinsic and extrinsic camera parameters
    if s.use_fisheye:
        _, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.fisheye.calibrate(object_points, image_points, image_size, camera_matrix, dist_coeffs) 
        rvecs = [_rvecs[i, :] for i in range(len(object_points))]
        tvecs = [_tvecs[i, :] for i in range(len(object_points))]
    else:
        _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs)

    success = cv2.checkRange(camera_matrix) and cv2.checkRange(dist_coeffs)

    total_average_errors, _ = compute_reprojection_err(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs, s.use_fisheye)

    # Find intrinsic and extrinsic camera parameters
    return success, camera_matrix, dist_coeffs, total_average_errors


def calculate_and_save_calibration_values(device):

    # Get initial node values ----------------------------------------------------------
    nodemap = device.nodemap
    acquisition_mode_initial = nodemap.get_node('AcquisitionMode').value
    pixel_format_initial = nodemap.get_node('PixelFormat').value

    # Set stream node values ----------------------------------------------------------
    tl_stream_nodemap = device.tl_stream_nodemap
    # Enable stream auto negotiate packet size
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    # Enable stream packet resend
    tl_stream_nodemap['StreamPacketResendEnable'].value = True
    
    # Set device node values ---------------------------------------------------------
    # Set resolution to match streaming resolution
    print(f'{TAB1}Set resolution to 2048x1536')
    try:
        nodemap.get_node('Width').value = 2048
    except Exception as e:
        print(f"{TAB1}  Warning: Could not set Width: {e}")
    try:
        nodemap.get_node('Height').value = 1536
    except Exception as e:
        print(f"{TAB1}  Warning: Could not set Height: {e}")
    
    # Set pixel format
    print(f'{TAB1}Set pixel format to \'Mono8\'')
    nodemap.get_node('PixelFormat').value = enums.PixelFormat.Mono8
    
    # Configure exposure for calibration (reset from any previous manual settings)
    print(f'{TAB1}Configure exposure for calibration')
    try:
        nodemap.get_node('ExposureAuto').value = 'Continuous'
        print(f'{TAB1}  Enabled auto exposure')
    except Exception as e:
        print(f'{TAB1}  Could not enable auto exposure: {e}')
        # Fallback: set high manual exposure
        try:
            nodemap.get_node('ExposureAuto').value = 'Off'
            nodemap.get_node('ExposureTime').value = 20000.0  # 20ms for bright image
            print(f'{TAB1}  Set manual exposure to 20ms')
        except Exception as e2:
            print(f'{TAB1}  Warning: Could not configure exposure: {e2}')
    
    # Set acquisitiom mode
    print(f'{TAB1}Set acquisition mode to \'Continuous\'')
    nodemap.get_node('AcquisitionMode').value = 'Continuous'
    
    # Set buffer handling mode
    print(f'{TAB1}Set bugfer handling mode to \'NewestOnly\'')
    tl_stream_nodemap['StreamBufferHandlingMode'].value = 'NewestOnly'

    # Start stream -----------------------------------------------------------------
    device.start_stream()

    # Get sets of calibration points ----------------------------------------------------
    print(f'{TAB1}Getting {NUM_IMAGES} sets of calibration points')
    print(f'{TAB1}Move the calibration target around the frame for best results')
    
    # Initialize values
    calibration_points = []
    image_size = [0] * 2
    attempts = 0
    images = 0
    grid_centers_found = 0
    successes = 0
    
    # Iterate until we get a sufficient number of images
    while successes < NUM_IMAGES:
        try:
            attempts += 1

            # get image
            buffer = device.get_buffer()
            images += 1
            
            if buffer.is_incomplete:
                raise RuntimeError('Incomplete image')

            # Copy image buffer into an OpenCV matrix (a numpy array) 
            buffer_Mono8 = convert_buffer_to_Mono8(buffer)
            buffer_bytes_per_pixel = int(len(buffer_Mono8.data)/(buffer_Mono8.width * buffer_Mono8.height))
            image_matrix = np.asarray(buffer_Mono8.data, dtype=np.uint8)
            image_matrix_reshaped = image_matrix.reshape(buffer_Mono8.height, buffer_Mono8.width, buffer_bytes_per_pixel)

            # OpenCV calibrateCamera expects image_size as (width, height), not (height, width)
            image_size[0] = buffer_Mono8.width
            image_size[1] = buffer_Mono8.height

            device.requeue_buffer(buffer)

            # Find calibration circles
            points_found, grid_centers = find_calibration_points(image_matrix_reshaped)

            grid_centers_found = 0 if not points_found else len(grid_centers)

            # If 20 calibration points are found, success
            if grid_centers_found == 20:
                calibration_points.append(grid_centers)
                successes += 1
                
                # Show preview with detected grid
                preview = cv2.cvtColor(image_matrix_reshaped, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(preview, (5, 4), grid_centers, True)
                # Add text overlay
                cv2.putText(preview, f'Calibration {successes}/{NUM_IMAGES} - Move target to a new position', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Calibration Preview', preview)
                # Non-blocking display - let the loop continue
                for _ in range(5):
                    cv2.waitKey(100)  # 5 x 100ms = 500ms total, but non-blocking
                
                # Skip a few frames to give user time to move target
                if successes < NUM_IMAGES:
                    print(f'{TAB2}Captured {successes}/{NUM_IMAGES}. Move target to a new position...')
                    for _ in range(2):  # Skip ~2 seconds of frames
                        buffer = device.get_buffer()
                        device.requeue_buffer(buffer)
                        time.sleep(1)
            
        except Exception as e:
            print(f'{TAB1}Exception [{e}] happened. Retry')
            
        print(f'{TAB2}{attempts} attempts, '
              f'{images} images, {grid_centers_found} circles found, '
              f'{successes} calibration points', end='\r')
        
        time.sleep(SLEEP_SECOND)
    
    print(f'{TAB2}{attempts} attempts, '
              f'{images} images, {grid_centers_found} circles found, '
              f'{successes} calibration points')
    
    # Close preview window
    cv2.destroyAllWindows()
    
    # Calculate camera matrix and distance coefficients -------------------------------------------------------------------
    print(f'{TAB1}Calculate camera matrix and distance coefficients')
    
    s = Settings()
    s.nr_frames = NUM_IMAGES
    s.input_type = Settings.InputType.IMAGE_LIST

    # Calculate the calibration values
    calculation_succeeded, camera_matrix, dist_coeffs, total_average_error = calculate(s, image_size, np.array(calibration_points))
    
    print(f'{TAB1}Calibration succeeded' if calculation_succeeded else f'{TAB2}Calibration failed')
    print(f'{TAB1}Calculated reprojection error is {total_average_error}')
    
    # Save calibration information ---------------------------------------------------------------
    print(f'{TAB1}Save camera matrix and distance coefficients to file {FILE_NAME}')    
    fs = cv2.FileStorage(FILE_NAME, cv2.FileStorage_WRITE)
    fs.write('cameraMatrix', camera_matrix)
    fs.write('distCoeffs', dist_coeffs)
    fs.release()
    
    # Stop stream
    device.stop_stream()
    
    # return nodes to their initial values
    nodemap.get_node('PixelFormat').value = pixel_format_initial
    nodemap.get_node('AcquisitionMode').value = acquisition_mode_initial


def example_entry_point():

    print(f'{TAB1}py_HLTRGB_1_Calibration')

    # Get connected devices ---------------------------------------------------

    # Create devices
    devices = create_devices_with_tries()

    # Filter applicable devices
    devices = get_applicable_devices(devices)

    # Select a Triton camera to use
    device = system.select_device(devices)

    # Calculate and save calibration values ----------------------------------------
    calculate_and_save_calibration_values(device)

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
