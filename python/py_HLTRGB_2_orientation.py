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
    TRITON, HELIOS2, TAB1, TAB2,
    create_devices_with_tries,
    get_applicable_devices,
    convert_buffer_to_Coord3D_ABCY16,
    convert_buffer_to_Mono8,
    extract_triton_mono8,
    read_calibration,
    write_orientation,
)
from common.node_overrides import CameraOverrides, safe_set_node, safe_get_node_value

TIMEOUT = 2000
FILE_NAME_IN = 'tritoncalibration.yml'
FILE_NAME_OUT = "orientation.yml"


def configure_helios_for_calibration(device):
    '''
    Configure Helios2 with aggressive settings optimized for calibration target detection.
    Returns: (xyz_scale_mm, x_offset_mm, y_offset_mm, z_offset_mm, saved_settings_dict)
    '''
    tl_stream_nodemap = device.tl_stream_nodemap
    safe_set_node(tl_stream_nodemap, 'StreamAutoNegotiatePacketSize', True)
    safe_set_node(tl_stream_nodemap, 'StreamPacketResendEnable', True)

    nodemap = device.nodemap
    safe_set_node(nodemap, 'PixelFormat', PixelFormat.Coord3D_ABCY16)
    
    # Save original settings to restore later

    saved_settings = {}
    
    print(f'{TAB1}Configuring Helios for calibration (aggressive settings)...')
    
    # 1. Maximize exposure time for better depth on white surfaces
    orig_exposure_sel = safe_get_node_value(nodemap, 'ExposureTimeSelector', None)
    if safe_set_node(nodemap, 'ExposureTimeSelector', 'Exp250Us'):
        if orig_exposure_sel is not None:
            saved_settings['ExposureTimeSelector'] = orig_exposure_sel
        print(f'{TAB1}  ExposureTimeSelector: {orig_exposure_sel} -> Exp250Us')
    else:
        print(f'{TAB1}  Info: Could not set exposure time')

    # 3. Lower confidence threshold to accept more pixels
    orig_conf = safe_get_node_value(nodemap, 'Scan3dConfidenceThreshold', None)
    conf_new = 50.0
    if safe_set_node(nodemap, 'Scan3dConfidenceThreshold', conf_new):
        if orig_conf is not None:
            saved_settings['Scan3dConfidenceThreshold'] = orig_conf
        print(f'{TAB1}  Scan3dConfidenceThreshold: {orig_conf} -> {conf_new}')
    else:
        print(f'{TAB1}  Info: Scan3dConfidenceThreshold not available')
    
    # 4. Increase accumulation/averaging if available
    orig_accum = safe_get_node_value(nodemap, 'Scan3dImageAccumulation', None)
    accum_new = 4
    if safe_set_node(nodemap, 'Scan3dImageAccumulation', accum_new):
        if orig_accum is not None:
            saved_settings['Scan3dImageAccumulation'] = orig_accum
        print(f'{TAB1}  Scan3dImageAccumulation: {orig_accum} -> {accum_new}')
    
    # 5. Disable any filtering that might reject valid pixels
    orig_spatial = safe_get_node_value(nodemap, 'Scan3dSpatialFilter', None)
    if safe_set_node(nodemap, 'Scan3dSpatialFilter', False):
        if orig_spatial is not None:
            saved_settings['Scan3dSpatialFilter'] = orig_spatial
        print(f'{TAB1}  Scan3dSpatialFilter: {orig_spatial} -> False')

    print(f'{TAB1}Get xyz coordinate scales and offsets from nodemap')
    xyz_scale_mm = safe_get_node_value(nodemap, 'Scan3dCoordinateScale', None)
    safe_set_node(nodemap, 'Scan3dCoordinateSelector', 'CoordinateA')
    x_offset_mm = safe_get_node_value(nodemap, 'Scan3dCoordinateOffset', None)
    safe_set_node(nodemap, 'Scan3dCoordinateSelector', 'CoordinateB')
    y_offset_mm = safe_get_node_value(nodemap, 'Scan3dCoordinateOffset', None)
    safe_set_node(nodemap, 'Scan3dCoordinateSelector', 'CoordinateC')
    z_offset_mm = safe_get_node_value(nodemap, 'Scan3dCoordinateOffset', None)

    return xyz_scale_mm, x_offset_mm, y_offset_mm, z_offset_mm, saved_settings


def restore_helios_settings(device, saved_settings):
    '''
    Restore Helios settings from saved_settings dict.
    '''
    if not saved_settings:
        return
    
    nodemap = device.nodemap
    print(f'{TAB1}Restoring Helios to original settings...')
    
    for param_name, original_value in saved_settings.items():
        if safe_set_node(nodemap, param_name, original_value):
            print(f'{TAB1}  {param_name}: restored to {original_value}')
        else:
            print(f'{TAB1}  Warning: Could not restore {param_name}')


def extract_helios_xyz_intensity(buffer, xyz_scale_mm, x_offset_mm, y_offset_mm, z_offset_mm):
    '''
    Given a Coord3D_ABCY16 buffer, return (intensity_image, xyz_mm).
    '''
    buffer_Coord3D_ABCY16 = convert_buffer_to_Coord3D_ABCY16(buffer)

    height = int(buffer_Coord3D_ABCY16.height)
    width = int(buffer_Coord3D_ABCY16.width)
    channels_per_pixel = int(buffer_Coord3D_ABCY16.bits_per_pixel / 16)

    if channels_per_pixel != 4:
        raise RuntimeError(f"Unexpected channel count for Coord3D_ABCY16: {channels_per_pixel}")

    ptr = ctypes.cast(buffer_Coord3D_ABCY16.pdata, ctypes.POINTER(ctypes.c_uint16))
    total_vals = height * width * channels_per_pixel
    raw = np.ctypeslib.as_array(ptr, shape=(total_vals,))
    pixels = raw.reshape(height * width, channels_per_pixel)

    A = pixels[:, 0].astype(np.float32).reshape(height, width)
    B = pixels[:, 1].astype(np.float32).reshape(height, width)
    C = pixels[:, 2].astype(np.float32).reshape(height, width)
    intensity_image = pixels[:, 3].reshape(height, width).astype(np.uint16)

    # Debug: Check for invalid data (sentinel value 32767 = 0x7FFF for signed coords)
    center_y, center_x = height // 2, width // 2
    raw_center = pixels[center_y * width + center_x, :]
    print(f'{TAB1}DEBUG Helios center pixel raw: A={raw_center[0]}, B={raw_center[1]}, C={raw_center[2]}, Intensity={raw_center[3]}', flush=True)
    print(f'{TAB1}DEBUG Helios scale/offsets: scale={xyz_scale_mm}, X_off={x_offset_mm}, Y_off={y_offset_mm}, Z_off={z_offset_mm}', flush=True)
    
    # Count valid vs invalid pixels
    invalid_mask = (pixels[:, 2] == 32767) | (pixels[:, 2] == 65535) | (pixels[:, 2] == 0)
    num_invalid = np.sum(invalid_mask)
    pct_invalid = 100.0 * num_invalid / len(pixels)
    print(f'{TAB1}DEBUG Helios invalid pixels: {num_invalid}/{len(pixels)} ({pct_invalid:.1f}%)', flush=True)

    X_mm = A * xyz_scale_mm + x_offset_mm
    Y_mm = B * xyz_scale_mm + y_offset_mm
    Z_mm = C * xyz_scale_mm + z_offset_mm
    xyz_mm = np.stack((X_mm, Y_mm, Z_mm), axis=2)

    return intensity_image, xyz_mm


def get_image_HLT(device):
    '''
    Returns intensity and xyz images from a Helios2 device (legacy function - not used in main calibration flow)
    '''
    xyz_scale_mm, x_offset_mm, y_offset_mm, z_offset_mm, _ = configure_helios_for_calibration(device)

    device.start_stream()
    buffer = device.get_buffer()

    intensity_image, xyz_mm = extract_helios_xyz_intensity(buffer, xyz_scale_mm, x_offset_mm, y_offset_mm, z_offset_mm)

    device.requeue_buffer(buffer)
    device.stop_stream()

    return intensity_image, xyz_mm

 


def get_image_TRI(device):
    '''
    Returns image from a Triton device
    '''
    configure_triton_for_capture(device)

    device.start_stream()
    buffer = device.get_buffer()

    image_matrix_reshaped = extract_triton_mono8(buffer)

    device.requeue_buffer(buffer)
    device.stop_stream()

    return image_matrix_reshaped


def configure_triton_for_capture(device):
    '''
    Configure Triton to match calibration stream settings.
    '''
    nodemap = device.nodemap
    # Ensure parameters are writable and ROI is valid before resizing
    # Unlock if supported
    safe_set_node(nodemap, 'TLParamsLocked', 0)

    # Reset binning/decimation to avoid alignment issues when expanding ROI
    for node_name, value in (
        ('BinningSelector', 'All'),
        ('BinningHorizontal', 1),
        ('BinningVertical', 1),
        ('DecimationHorizontal', 1),
        ('DecimationVertical', 1),
    ):
        safe_set_node(nodemap, node_name, value)

    # Reset offsets to origin so width/height can be maximized safely
    for node_name in ('OffsetX', 'OffsetY'):
        safe_set_node(nodemap, node_name, 0)

    # Helper to align to node constraints
    def _align_to_node(node, desired):
        try:
            nmin = getattr(node, 'min')
            nmax = getattr(node, 'max')
            inc = getattr(node, 'inc', None)
            if inc is None:
                inc = getattr(node, 'increment', 1)
            inc = 1 if inc in (None, 0) else inc
            val = max(nmin, min(desired, nmax))
            if inc > 1:
                val = nmin + ((val - nmin) // inc) * inc
            return int(val)
        except Exception:
            return int(desired)

    # Set width/height with alignment and fallbacks
    try:
        width_node = nodemap.get_node('Width')
        height_node = nodemap.get_node('Height')
        desired_w = 2048
        desired_h = 1536
        w = _align_to_node(width_node, desired_w)
        h = _align_to_node(height_node, desired_h)
        # Some devices require height before width; try height first
        try:
            # Prefer safe_set_node for consistency
            set1 = safe_set_node(nodemap, 'Height', h) and safe_set_node(nodemap, 'Width', w)
            if not set1:
                # Fallback to direct if wrappers fail silently
                height_node.value = h
                width_node.value = w
        except Exception:
            # Try the opposite order
            set2 = safe_set_node(nodemap, 'Width', w) and safe_set_node(nodemap, 'Height', h)
            if not set2:
                width_node.value = w
                height_node.value = h
    except Exception as e:
        print(f"{TAB1}Warning: Could not set Triton ROI to 2048x1536: {e}")

    # Prefer RGB8 for easier processing downstream (we convert to Mono8 later)
    if not safe_set_node(nodemap, 'PixelFormat', PixelFormat.RGB8):
        print(f"{TAB1}Warning: Could not set PixelFormat RGB8")
    
    # Add exposure control to prevent IR overexposure from Helios
    if safe_set_node(nodemap, 'ExposureAuto', 'Off') and safe_set_node(nodemap, 'ExposureTime', 250):
        print(f'{TAB1}Set Triton exposure to 250 us (manual)')
    else:
        print(f'{TAB1}Warning: Could not set Triton exposure')
    
    # Add gain to brighten the image
    if safe_set_node(nodemap, 'GainAuto', 'Off') and safe_set_node(nodemap, 'Gain', 6.0):
        print(f'{TAB1}Set Triton gain to 6.0 dB')

    tl_stream_nodemap = device.tl_stream_nodemap
    safe_set_node(tl_stream_nodemap, 'StreamAutoNegotiatePacketSize', True)
    safe_set_node(tl_stream_nodemap, 'StreamPacketResendEnable', True)
    safe_set_node(tl_stream_nodemap, 'StreamBufferHandlingMode', 'NewestOnly')


 


def capture_synced_frames(device_triton, device_helios2):
    '''
    Start both cameras, grab frames in quick succession, and return aligned data.
    CRITICAL: Capture Triton FIRST to avoid IR contamination from Helios projector.
    Returns: (intensity_image, xyz_mm, image_matrix_TRI, helios_saved_settings)
    '''
    with CameraOverrides(
        device_triton,
        nodes={k: v for k, v in {
            'PixelFormat': safe_get_node_value(device_triton.nodemap, 'PixelFormat', None),
        }.items() if v is not None},
        stream_nodes={k: v for k, v in {
            'StreamAutoNegotiatePacketSize': safe_get_node_value(device_triton.tl_stream_nodemap, 'StreamAutoNegotiatePacketSize', None),
            'StreamPacketResendEnable': safe_get_node_value(device_triton.tl_stream_nodemap, 'StreamPacketResendEnable', None),
            'StreamBufferHandlingMode': safe_get_node_value(device_triton.tl_stream_nodemap, 'StreamBufferHandlingMode', None),
        }.items() if v is not None}
    ), CameraOverrides(
        device_helios2,
        nodes={k: v for k, v in {
            'PixelFormat': safe_get_node_value(device_helios2.nodemap, 'PixelFormat', None),
        }.items() if v is not None},
        stream_nodes={k: v for k, v in {
            'StreamAutoNegotiatePacketSize': safe_get_node_value(device_helios2.tl_stream_nodemap, 'StreamAutoNegotiatePacketSize', None),
            'StreamPacketResendEnable': safe_get_node_value(device_helios2.tl_stream_nodemap, 'StreamPacketResendEnable', None),
            'StreamBufferHandlingMode': safe_get_node_value(device_helios2.tl_stream_nodemap, 'StreamBufferHandlingMode', None),
        }.items() if v is not None}
    ):
        # Configure with aggressive calibration-optimized settings
        xyz_scale_mm, x_offset_mm, y_offset_mm, z_offset_mm, helios_saved_settings = configure_helios_for_calibration(device_helios2)
        configure_triton_for_capture(device_triton)

        buffer_h = None
        buffer_t = None

        # Start Triton FIRST and let it stabilize
        device_triton.start_stream()
        
        try:
            # Warm-up Triton first (2-3 frames to stabilize exposure)
            print(f'{TAB1}Warming up Triton camera...')
            for _ in range(3):
                tmp = device_triton.get_buffer()
                device_triton.requeue_buffer(tmp)
                time.sleep(0.1)  # Let exposure stabilize
            
            # Capture Triton frame BEFORE starting Helios (avoids IR contamination)
            print(f'{TAB1}Capturing Triton frame (before Helios IR)...')
            buffer_t = device_triton.get_buffer()
            image_matrix_TRI = extract_triton_mono8(buffer_t)
            device_triton.requeue_buffer(buffer_t)
            buffer_t = None
            
            # Stop Triton now that we have clean RGB data
            device_triton.stop_stream()
            
            # NOW start Helios and capture depth (IR won't affect Triton anymore)
            print(f'{TAB1}Capturing Helios frame...')
            device_helios2.start_stream()
            
            # Warm-up Helios
            for _ in range(2):
                tmp = device_helios2.get_buffer()
                device_helios2.requeue_buffer(tmp)

            buffer_h = device_helios2.get_buffer()
            intensity_image, xyz_mm = extract_helios_xyz_intensity(buffer_h, xyz_scale_mm, x_offset_mm, y_offset_mm, z_offset_mm)
            
        finally:
            if buffer_h is not None:
                device_helios2.requeue_buffer(buffer_h)
            if buffer_t is not None:
                device_triton.requeue_buffer(buffer_t)
            device_helios2.stop_stream()
            # Triton already stopped above

        return intensity_image, xyz_mm, image_matrix_TRI, helios_saved_settings

def find_calibration_points_HLT(image_in_orig):
    '''
    Returns an array of calibration points found in the given image captured by Helios2
    '''

    # Normalize to 8-bit and gently blur to stabilize blob detection
    _, max_value, _, _ = cv2.minMaxLoc(image_in_orig)
    if max_value <= 0:
        return False, None
    image_8bit = cv2.convertScaleAbs(image_in_orig, alpha=255.0/max_value)
    image_8bit = cv2.GaussianBlur(image_8bit, (5, 5), 0)

    def try_detect(image_u8, blob_color):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = blob_color
        params.thresholdStep = 5
        params.minThreshold = 10
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 5.0
        params.maxArea = 10000.0
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        params.filterByConvexity = True
        params.minConvexity = 0.7
        detector = cv2.SimpleBlobDetector.create(params)
        pattern_size = (5, 4)
        found, centers = cv2.findCirclesGrid(
            image_u8,
            pattern_size,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=detector,
        )
        return found, centers

    # Try bright (white) circles first, then dark (black) circles
    found, centers = try_detect(image_8bit, 255)
    if not found:
        found, centers = try_detect(image_8bit, 0)

    return found, centers


def find_calibration_points_TRI(image_in_orig):
    '''
    Returns an array of calibration points found in the given image captured by Triton
    '''
    print(f'{TAB2}Starting TRI calibration point detection...', flush=True)

    scaling = 1.0
    image_in = image_in_orig
    num_cols_orig = image_in_orig.shape[1] # width
    num_rows_orig = image_in_orig.shape[0] # height

    pattern_size = (5, 4) # (pattern_per_row, pattern_per_column)

    def try_detect(image_u8, blob_color):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = blob_color
        params.filterByCircularity = True
        params.minCircularity = 0.5  # More relaxed for softer circles
        params.filterByArea = True
        params.minArea = 8.0  # Lower minimum
        params.maxArea = 8000.0  # Higher maximum for high-res
        params.filterByInertia = True
        params.minInertiaRatio = 0.2  # More relaxed
        params.filterByConvexity = True
        params.minConvexity = 0.5  # More relaxed
        params.thresholdStep = 8  # Finer steps for better detection
        params.minThreshold = 10  # Lower min threshold
        params.maxThreshold = 250  # Higher max threshold
        detector = cv2.SimpleBlobDetector.create(params)
        
        # Add timeout hint via small iteration count
        try:
            return cv2.findCirclesGrid(
                image_u8,
                pattern_size,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                blobDetector=detector,
            )
        except Exception as e:
            print(f' [error: {e}]', flush=True)
            return False, None

    # Enhance image contrast and sharpness before detection
    # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image_in_orig = clahe.apply(image_in_orig)
    
    # 2. Optional: Sharpen the image to make circles crisper
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    image_in_orig = cv2.filter2D(image_in_orig, -1, kernel_sharpen)
    
    # 3. Light blur to reduce noise after sharpening
    image_in_orig = cv2.GaussianBlur(image_in_orig, (3, 3), 0)

    try:
        # Start with downsampled versions first (much faster!)
        # Work from small to large to avoid hanging on huge images
        scales_to_try = [4.0, 2.0, 1.5, 1.0]  # Start small, work up to full-res
        
        for scale_factor in scales_to_try:
            scaling = scale_factor
            target_h = int(num_rows_orig / scaling)
            target_w = int(num_cols_orig / scaling)
            
            print(f'{TAB2}Trying {target_w}x{target_h} (scale {scale_factor:.1f}x)...', end='', flush=True)
            
            if scale_factor == 1.0:
                image_in = image_in_orig
            else:
                image_in = cv2.resize(image_in_orig, (target_w, target_h))
            
            found, centers = try_detect(image_in, 255)
            if not found:
                found, centers = try_detect(image_in, 0)
            
            if found:
                print(' FOUND!', flush=True)
                break
            else:
                print(' not found', flush=True)
    except KeyboardInterrupt:
        print(f'\n{TAB2}Detection cancelled by user', flush=True)
        return False, None

    if found:
        for center in centers:
            center[0][0] *= scaling
            center[0][1] *= scaling
    else:
        print(f'{TAB2}Failed to detect grid at any scale')

    return found, centers
    

'''
EXAMPLE
'''
def calculate_and_save_orientation_values(device_triton, device_helios2):
    
    # Read in camera matrix and distance coefficient ----------------------------------------------------------------
    print(f'{TAB1}Read camera matrix and distance coefficients from file {FILE_NAME_IN}')
    camera_matrix, dist_coeffs = read_calibration(FILE_NAME_IN)

    # Capture a synchronized pair of frames ---------------------------------------------------
    print(f'{TAB1}Capture synchronized frames from both cameras')
    image_matrix_HLT_intensity, image_matrix_HLT_XYZ, image_matrix_TRI, helios_saved_settings = capture_synced_frames(device_triton, device_helios2)

    # Visualize Helios depth data ---------------------------------------------------
    print(f'{TAB1}Visualizing Helios depth data...')
    depth_mm = image_matrix_HLT_XYZ[:, :, 2]  # Z channel = depth
    
    # Mark invalid pixels
    valid_mask = (depth_mm > 0) & (depth_mm < 10000)  # Valid depth between 0-10m
    
    if np.any(valid_mask):
        depth_viz = depth_mm.copy()
        depth_viz[~valid_mask] = np.nan  # Set invalid to NaN for visualization
        
        # Normalize valid depths for colormap
        valid_depths = depth_viz[valid_mask]
        dmin = float(np.min(valid_depths))
        dmax = float(np.max(valid_depths))
        
        # Create heatmap
        depth_norm = np.zeros_like(depth_viz, dtype=np.uint8)
        depth_norm[valid_mask] = np.clip(((depth_viz[valid_mask] - dmin) / (dmax - dmin) * 255.0), 0, 255).astype(np.uint8)
        depth_heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        # Mark invalid pixels in gray
        depth_heatmap[~valid_mask] = (50, 50, 50)
        
        # Add info overlay
        cv2.putText(depth_heatmap, f'Helios Depth Map', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_heatmap, f'Range: {dmin:.0f}-{dmax:.0f} mm', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        valid_pct = 100.0 * np.sum(valid_mask) / valid_mask.size
        cv2.putText(depth_heatmap, f'Valid: {valid_pct:.1f}%', (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_heatmap, 'Gray = No Depth', (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow("Helios Depth Map (Z channel)", depth_heatmap)
        cv2.waitKey(1)
    
    # Calculate orientation values ----------------------------------------------------------------
    print(f'{TAB1}Calculate orientation values')

    # Find HLT calibration points using HLT intensity image
    print(f'{TAB1}Find points in HLT image')
    count = 0
    hlt_intensity_8u = cv2.normalize(image_matrix_HLT_intensity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    tri_intensity_8u = cv2.normalize(image_matrix_TRI, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if tri_intensity_8u.shape != hlt_intensity_8u.shape:
        tri_intensity_8u = cv2.resize(tri_intensity_8u, (hlt_intensity_8u.shape[1], hlt_intensity_8u.shape[0]))

    cv2.imshow("HLT Intensity (8-bit normalized)", hlt_intensity_8u)
    cv2.imshow("TRI Intensity (8-bit normalized)", tri_intensity_8u)

    cv2.waitKey(1)
    while True:
        grid_centers_HLT_found, grid_centers_HLT = find_calibration_points_HLT(image_matrix_HLT_intensity)
        if not grid_centers_HLT_found or len(grid_centers_HLT) != 20:
            print(f'{TAB2}Unable to find points in HLT intensity image. {count} seconds passed', end='\r')
            count += 1
            time.sleep(1)
        else:
            # Show detected grid on HLT intensity image
            hlt_preview = cv2.cvtColor(hlt_intensity_8u, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(hlt_preview, (5, 4), grid_centers_HLT, True)
            cv2.putText(hlt_preview, 'HLT: 20 points detected!', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("HLT Detection Result", hlt_preview)
            
            # Also overlay circles on depth map to show which have depth
            if 'depth_heatmap' in locals():
                depth_with_circles = depth_heatmap.copy()
                for i, pt in enumerate(grid_centers_HLT):
                    c1 = int(round(pt[0][0]))
                    r1 = int(round(pt[0][1]))
                    z_val = depth_mm[r1, c1] if 0 <= r1 < depth_mm.shape[0] and 0 <= c1 < depth_mm.shape[1] else 0
                    has_depth = (z_val > 0 and z_val < 10000)
                    color = (0, 255, 0) if has_depth else (0, 0, 255)
                    cv2.circle(depth_with_circles, (c1, r1), 8, color, 2)
                    cv2.putText(depth_with_circles, str(i), (c1+10, r1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(depth_with_circles, 'Green=Valid Depth, Red=No Depth', (10, depth_with_circles.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Helios Depth Map with Circles", depth_with_circles)
            
            cv2.waitKey(2000)  # Show for 2 seconds
            print(f'\n{TAB2}Successfully detected 20 points in HLT image')
            break

    # Find TRI calibration points
    print(f'{TAB1}Find points in TRI image')
    print(f'{TAB2}(Press Ctrl+C to abort if detection fails)')
    
    try:
        # Only try detection once - if it fails after all the downsampling attempts, user needs to reposition target
        grid_centers_TRI_found, grid_centers_TRI = find_calibration_points_TRI(image_matrix_TRI)
        
        # Debug: show what we got
        num_points = len(grid_centers_TRI) if grid_centers_TRI is not None else 0
        print(f"\n{TAB2}Detection result: found={grid_centers_TRI_found}, points={num_points}", flush=True)
        
        if not grid_centers_TRI_found or (grid_centers_TRI is not None and len(grid_centers_TRI) != 20):
            print(f"\n{TAB2}ERROR: Could not detect 20 calibration points in TRI image!")
            print(f"{TAB2}Detected {num_points} points (need exactly 20)")
            print(f"{TAB2}Please check:")
            print(f"{TAB2}  - Target is visible and in focus in the 'TRI Intensity' window")
            print(f"{TAB2}  - Target has good lighting/contrast")
            print(f"{TAB2}  - Target is at appropriate distance")
            print(f"{TAB2}  - All 20 circles are visible in frame")
            print(f"{TAB2}Reposition target and re-run the script.")
            
            # Save the image for debugging
            debug_filename = 'tri_detection_failed.png'
            cv2.imwrite(debug_filename, tri_intensity_8u)
            print(f"{TAB2}Saved TRI image to '{debug_filename}' for inspection")
            
            # Show what blobs were detected (if any) for diagnosis
            if grid_centers_TRI is not None and num_points > 0:
                tri_original_normalized = cv2.normalize(image_matrix_TRI, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                debug_vis = cv2.cvtColor(tri_original_normalized, cv2.COLOR_GRAY2BGR)
                for pt in grid_centers_TRI:
                    center = (int(pt[0][0]), int(pt[0][1]))
                    cv2.circle(debug_vis, center, 8, (0, 255, 0), 2)
                cv2.putText(debug_vis, f'Detected {num_points} points (need 20)', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("TRI Detection Failed - Partial Result", debug_vis)
            
            print(f"{TAB2}Windows left open for inspection. Close them manually when done.")
            return
        
        # Show detected grid on TRI image (use original image_matrix_TRI, not the resized tri_intensity_8u)
        tri_original_normalized = cv2.normalize(image_matrix_TRI, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tri_preview = cv2.cvtColor(tri_original_normalized, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(tri_preview, (5, 4), grid_centers_TRI, True)
        cv2.putText(tri_preview, 'TRI: 20 points detected!', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("TRI Detection Result", tri_preview)
        cv2.waitKey(2000)  # Show for 2 seconds
        print(f'\n{TAB2}Successfully detected 20 points in TRI image')
        
    except KeyboardInterrupt:
        print(f'\n{TAB2}Detection cancelled by user')
        print(f"{TAB2}Windows left open for inspection. Close them manually when done.")
        return

    # Show combined view of both detections
    hlt_small = cv2.resize(hlt_preview, (hlt_preview.shape[1]//2, hlt_preview.shape[0]//2))
    tri_small = cv2.resize(tri_preview, (tri_preview.shape[1]//2, tri_preview.shape[0]//2))
    # Make heights match
    if hlt_small.shape[0] < tri_small.shape[0]:
        hlt_small = cv2.resize(hlt_small, (hlt_small.shape[1], tri_small.shape[0]))
    elif tri_small.shape[0] < hlt_small.shape[0]:
        tri_small = cv2.resize(tri_small, (tri_small.shape[1], hlt_small.shape[0]))
    combined = np.hstack((hlt_small, tri_small))
    cv2.putText(combined, 'Both cameras: grids detected! Computing orientation...', 
               (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Orientation - Combined View", combined)
    cv2.waitKey(1500)

    # Prepare for PnP
    print(f'{TAB1}Prepare for PnP')

    target_points_3d_mm = []
    target_points_3d_pixels = []
    target_points_2d_pixels = []


    num_invalid_circles = 0
    for i in range(len(grid_centers_TRI)):
        c1 = round(grid_centers_HLT[i][0][0])
        r1 = round(grid_centers_HLT[i][0][1])
        c2 = round(grid_centers_TRI[i][0][0])
        r2 = round(grid_centers_TRI[i][0][1])

        [x, y, z] = [image_matrix_HLT_XYZ[r1, c1][j] for j in range(3)]
        pt = [x, y, z]
        
        # Check if this circle has invalid depth (sentinel values)
        is_invalid = (abs(x) > 8000 and abs(y) > 8000 and z > 16000)
        if is_invalid:
            num_invalid_circles += 1
        
        if i < 5 or i >= len(grid_centers_TRI) - 2:  # Print first 5 and last 2
            status = " [INVALID DEPTH!]" if is_invalid else ""
            print(f'{TAB2}Point {i}: [{x:.1f}, {y:.1f}, {z:.1f}] mm{status}')

        target_points_3d_mm.append(pt)
        target_points_3d_pixels.append(grid_centers_HLT[i][0])
        target_points_2d_pixels.append(grid_centers_TRI[i][0])
    
    if num_invalid_circles > 15:
        print(f'\n{TAB2}ERROR: {num_invalid_circles}/20 calibration circles have NO VALID DEPTH!')
        print(f'{TAB2}The Helios cannot see depth at the circle locations.')
        print(f'{TAB2}')
        print(f'{TAB2}SOLUTION: Make the calibration target 3D:')
        print(f'{TAB2}  - Stick small objects (foam dots, coins, tape) on each circle')
        print(f'{TAB2}  - Or place textured material (cardboard/foam) right behind target')
        print(f'{TAB2}  - The circles need to have measurable depth, not just be flat paper')
        return

    target_points_3d_mm = np.array(target_points_3d_mm)
    target_points_3d_pixels = np.array(target_points_3d_pixels)
    target_points_2d_pixels = np.array(target_points_2d_pixels)
    
    # Sanity check: Are all XYZ points identical? (would cause solvePnP to fail)
    xyz_std = np.std(target_points_3d_mm, axis=0)
    print(f'{TAB2}XYZ standard deviation: [{xyz_std[0]:.1f}, {xyz_std[1]:.1f}, {xyz_std[2]:.1f}] mm')
    if np.all(xyz_std < 1.0):
        print(f'{TAB2}ERROR: All XYZ points are nearly identical! Cannot compute orientation.')
        print(f'{TAB2}Check that Helios2 is capturing valid depth data.')
        return
    
    object_points = target_points_3d_mm.astype(np.float64)
    image_points = target_points_2d_pixels.astype(np.float64)

    centroid_helios = np.mean(object_points, axis=0)

    def evaluate_solution(rvec, tvec, flag_name):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        centroid_cam = R @ centroid_helios.reshape(3, 1) + tvec
        z_centroid = centroid_cam[2, 0]
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points, axis=1)
        mean_err = float(np.mean(errors))
        return {
            'flag': flag_name,
            'rvec': rvec,
            'tvec': tvec,
            'z_centroid': float(z_centroid),
            'mean_error': mean_err
        }

    candidates = []

    # Try classic iterative solution
    try:
        success_iter, rvec_iter, tvec_iter = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success_iter:
            info = evaluate_solution(rvec_iter, tvec_iter, 'SOLVEPNP_ITERATIVE')
            candidates.append(info)
            if info['z_centroid'] <= 0:
                info_flip = evaluate_solution(rvec_iter, -info['tvec'], 'SOLVEPNP_ITERATIVE_FLIPPED')
                candidates.append(info_flip)
    except cv2.error as e:
        print(f"{TAB2}Warning: solvePnP iterative failed ({e})")

    # Try planar-specific solvers to disambiguate mirrored solutions
    planar_flags = [
        ('SOLVEPNP_IPPE', cv2.SOLVEPNP_IPPE),
        ('SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_IPPE_SQUARE),
        ('SOLVEPNP_EPNP', cv2.SOLVEPNP_EPNP),
        ('SOLVEPNP_AP3P', cv2.SOLVEPNP_AP3P),
    ]
    for flag_name, flag in planar_flags:
        try:
            success_generic, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=flag
            )
            if success_generic and len(rvecs):
                for idx, (rvec_candidate, tvec_candidate) in enumerate(zip(rvecs, tvecs)):
                    info = evaluate_solution(rvec_candidate, tvec_candidate, f'{flag_name}[{idx}]')
                    candidates.append(info)
                    if info['z_centroid'] <= 0:
                        info_flip = evaluate_solution(rvec_candidate, -info['tvec'], f'{flag_name}[{idx}]_FLIPPED')
                        candidates.append(info_flip)
        except cv2.error as e:
            print(f"{TAB2}Info: {flag_name} solver not available ({e})")

    if not len(candidates):
        print(f"{TAB1}ERROR: Unable to solve PnP for orientation")
        return

    positive_candidates = [c for c in candidates if c['z_centroid'] > 0]
    if positive_candidates:
        best = min(positive_candidates, key=lambda c: (c['mean_error'], -c['z_centroid']))
    else:
        best = min(candidates, key=lambda c: c['mean_error'])
        print(f"{TAB1}WARNING: All candidate poses place centroid behind Triton (best Z = {best['z_centroid']:.1f} mm). Using lowest-error pose anyway.")

    rotation_vector = best['rvec']
    translation_vector = best['tvec']

    print(f"{TAB1}Orientation solver selected: {best['flag']}")
    print(f"{TAB1}Centroid Z in Triton frame: {best['z_centroid']:.1f} mm")
    print(f"{TAB1}Mean reprojection error: {best['mean_error']:.2f} px")

    translation_magnitude = np.linalg.norm(translation_vector)
    print(f'{TAB1}Translation vector: [{translation_vector[0][0]:.1f}, {translation_vector[1][0]:.1f}, {translation_vector[2][0]:.1f}] mm')
    print(f'{TAB1}Translation magnitude: {translation_magnitude:.1f} mm')
    if translation_magnitude > 1000:
        print(f'{TAB1}WARNING: Translation is > 1 meter! This seems wrong.')
        print(f'{TAB1}Expected cameras to be within ~200mm of each other.')
        print(f'{TAB1}Continuing anyway, but results may be bad...')

    # Save orientation information ----------------------------------------------------------------
    print(f'{TAB1}Save camera matrix, distance coefficients, and rotation and translation vectors to file {FILE_NAME_OUT}')
    # Persist calibration image size for robust intrinsic scaling in live overlay
    tri_h, tri_w = image_matrix_TRI.shape[:2]
    write_orientation(FILE_NAME_OUT, camera_matrix, dist_coeffs, rotation_vector, translation_vector, image_size=(tri_w, tri_h))

    # Keep preview windows open for inspection
    print(f'\n{TAB1}All preview windows left open for inspection.')
    
    # Restore Helios to normal operating settings ----------------------------------------------------------------
    restore_helios_settings(device_helios2, helios_saved_settings)


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
    print('Windows are still open - press any key in a window to close all and exit.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()