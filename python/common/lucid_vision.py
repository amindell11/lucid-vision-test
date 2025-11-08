import time
from typing import Tuple, Any, Dict, Optional

import numpy as np
import cv2
from arena_api import enums
from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.enums import PixelFormat

# Formatting helpers kept consistent with examples
TAB1 = "  "
TAB2 = "    "

# Device labels
TRITON = 'Triton'
HELIOS2 = 'Helios2'


def create_devices_with_tries(tries_max: int = 6, sleep_time_secs: int = 10):
    tries = 0
    while tries < tries_max:
        devices = system.create_device()
        if not devices:
            print(f'{TAB1}Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} secs for a device to be connected!')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                print(f'{TAB1}{sec_count + 1 } seconds passed ', '.' * sec_count, end='\r')
            tries += 1
        else:
            print(f'{TAB1}Created {len(devices)} device(s)')
            return devices
    raise Exception(f'{TAB1}No device found! Please connect a device and run the example again.')


def _is_triton(device) -> bool:
    model_name = device.nodemap.get_node('DeviceModelName').value
    return "TRI" in model_name and "-C" in model_name


def _is_helios2(device) -> bool:
    model_name = device.nodemap.get_node('DeviceModelName').value
    return ("HLT" in model_name) or ("HTP" in model_name) or ("HTW" in model_name) or ("HT" in model_name)


def get_applicable_devices(devices, type_label: str):
    applicable_devices = []
    for device in devices:
        if type_label == TRITON and _is_triton(device):
            applicable_devices.append(device)
        elif type_label == HELIOS2 and _is_helios2(device):
            applicable_devices.append(device)
    if not len(applicable_devices):
        raise Exception(f'{TAB1}No applicable device found! Please connect an Triton and Helios2 device and run the example again.')
    print(f'{TAB1}Detected {len(applicable_devices)} applicable {type_label} device(s)')
    return applicable_devices


def convert_buffer_to_Mono8(buffer):
    if buffer.pixel_format == enums.PixelFormat.Mono8:
        return buffer
    print(f'{TAB1}Converting image buffer pixel format to Mono8 ')
    return BufferFactory.convert(buffer, enums.PixelFormat.Mono8)


def convert_buffer_to_Coord3D_ABCY16(buffer):
    if buffer.pixel_format == enums.PixelFormat.Coord3D_ABCY16:
        return buffer
    print(f'{TAB1}Converting image buffer pixel format to Coord3D_ABCY16')
    return BufferFactory.convert(buffer, enums.PixelFormat.Coord3D_ABCY16)


def extract_triton_mono8(buffer) -> np.ndarray:
    buf_mono = convert_buffer_to_Mono8(buffer)
    img = np.asarray(buf_mono.data, dtype=np.uint8)
    return img.reshape(buf_mono.height, buf_mono.width)


def get_helios_xyz_scale_offsets(nodemap) -> Tuple[float, float, float, float]:
    xyz_scale_mm = nodemap["Scan3dCoordinateScale"].value
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateA"
    x_offset_mm = nodemap["Scan3dCoordinateOffset"].value
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateB"
    y_offset_mm = nodemap["Scan3dCoordinateOffset"].value
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
    z_offset_mm = nodemap["Scan3dCoordinateOffset"].value
    return xyz_scale_mm, x_offset_mm, y_offset_mm, z_offset_mm


def configure_stream_defaults(tl_stream_nodemap, newest_only: bool = False):
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True
    if newest_only:
        try:
            tl_stream_nodemap['StreamBufferHandlingMode'].value = 'NewestOnly'
        except Exception:
            pass


def write_calibration(file_name: str, camera_matrix, dist_coeffs):
    fs = cv2.FileStorage(file_name, cv2.FileStorage_WRITE)
    fs.write('cameraMatrix', camera_matrix)
    fs.write('distCoeffs', dist_coeffs)
    fs.release()


def read_calibration(file_name: str):
    fs = cv2.FileStorage(file_name, cv2.FileStorage_READ)
    camera_matrix = fs.getNode('cameraMatrix').mat()
    dist_coeffs = fs.getNode('distCoeffs').mat()
    fs.release()
    return camera_matrix, dist_coeffs


def write_orientation(file_name: str, camera_matrix, dist_coeffs, rotation_vector, translation_vector, image_size=None):
    fs = cv2.FileStorage(file_name, cv2.FileStorage_WRITE)
    fs.write('cameraMatrix', camera_matrix)
    fs.write('distCoeffs', dist_coeffs)
    fs.write('rotationVector', rotation_vector)
    fs.write('translationVector', translation_vector)
    # Optionally store the calibration image size for later intrinsic scaling
    try:
        if image_size is not None and len(image_size) == 2:
            fs.write('calibratedImageWidth', int(image_size[0]))
            fs.write('calibratedImageHeight', int(image_size[1]))
    except Exception:
        # Best-effort: do not fail orientation write if size cannot be written
        pass
    fs.release()


def read_orientation(file_name: str):
    fs = cv2.FileStorage(file_name, cv2.FileStorage_READ)
    camera_matrix = fs.getNode('cameraMatrix').mat()
    dist_coeffs = fs.getNode('distCoeffs').mat()
    rotation_vector = fs.getNode('rotationVector').mat()
    translation_vector = fs.getNode('translationVector').mat()
    fs.release()
    return camera_matrix, dist_coeffs, rotation_vector, translation_vector


