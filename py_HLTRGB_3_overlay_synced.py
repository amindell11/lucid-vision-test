import time
import ctypes
from os.path import exists

import numpy as np  # pip3 install numpy
import cv2  # pip3 install opencv-python

from arena_api import enums
from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.enums import PixelFormat


# Formatting helpers
TAB1 = "  "
TAB2 = "    "

# Calibration/orientation file
ORIENTATION_FILE = 'orientation.yml'

# Device designators
TRITON = 'Triton'
HELIOS2 = 'Helios2'

# Depth heatmap fixed color scale (mm)
DEPTH_COLOR_MIN_MM = 300.0

DEPTH_COLOR_MAX_MM = 1550.0

'''
Device helpers
'''
def create_devices_with_tries():
    tries = 0
    tries_max = 6
    sleep_time_secs = 10
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


def is_applicable_device_triton(device):
    model_name = device.nodemap.get_node('DeviceModelName').value
    return "TRI" in model_name and "-C" in model_name


def is_applicable_device_helios2(device):
    model_name = device.nodemap.get_node('DeviceModelName').value
    return "HLT" in model_name or "HTP" in model_name or "HTW" in model_name


def get_applicable_devices(devices, type_label):
    applicable_devices = []
    for device in devices:
        if type_label == TRITON and is_applicable_device_triton(device):
            applicable_devices.append(device)
        elif type_label == HELIOS2 and is_applicable_device_helios2(device):
            applicable_devices.append(device)
    if not applicable_devices:
        raise Exception(f'{TAB1}No applicable device found! Please connect a Triton and Helios2 device and run the example again.')
    print(f'{TAB1}Detected {len(applicable_devices)} applicable {type_label} device(s)')
    return applicable_devices


'''
Streaming setup
'''
def enable_ptp(device, label):
    nm = device.nodemap
    enabled = False
    # Try PTP enable nodes used by different firmware variants
    try:
        nm.get_node('PtpEnable').value = True
        enabled = True
    except Exception:
        pass
    if not enabled:
        try:
            nm.get_node('GevIEEE1588').value = True
            enabled = True
        except Exception:
            pass
    # Auto role if available
    for node_name, auto_value in (
        ('PtpRole', 'Auto'),
        ('GevIEEE1588ClockControl', 'Auto'),
    ):
        try:
            nm.get_node(node_name).value = auto_value
        except Exception:
            pass
    # Wait briefly for status to stabilize
    status = None
    start = time.perf_counter()
    while (time.perf_counter() - start) < 5.0:
        try:
            status = nm.get_node('PtpStatus').value
        except Exception:
            try:
                status = nm.get_node('GevIEEE1588Status').value
            except Exception:
                status = None
        if status and (('Master' in status) or ('Slave' in status) or ('Locked' in status)):
            break
        time.sleep(0.2)
    print(f"{TAB1}{label} PTP: {'enabled' if enabled else 'not enabled'} | status: {status if status else 'unknown'}")
    return enabled

def setup_triton_for_stream(device_triton):
    nodemap = device_triton.nodemap
    # Prefer lower bandwidth: apply binning/decimation and Bayer format
    # Try binning first
    try:
        try:
            nodemap.get_node('BinningSelector').value = 'All'
        except Exception:
            pass
        nodemap.get_node('BinningHorizontal').value = 2
        nodemap.get_node('BinningVertical').value = 2
    except Exception:
        # Fallback to decimation if binning not available
        try:
            nodemap.get_node('DecimationHorizontal').value = 2
            nodemap.get_node('DecimationVertical').value = 2
        except Exception:
            pass

    # Use BayerRG8 (1 byte/pixel) to reduce bandwidth
    try:
        nodemap.get_node('PixelFormat').value = PixelFormat.BayerRG8
    except Exception:
        # Fallback to RGB8 if Bayer not available
        nodemap.get_node('PixelFormat').value = PixelFormat.RGB8

    # Disable link limit that caps frame rate based on bandwidth
    try:
        nodemap.get_node('AcquisitionFrameRateLinkLimitEnable').value = False
    except Exception:
        pass
    # Try multiple approaches to enable frame rate control
    fps_enabled = False
    try:
        nodemap.get_node('AcquisitionFrameRateEnable').value = True
        fps_enabled = True
    except Exception:
        pass
    if not fps_enabled:
        try:
            nodemap.get_node('AcquisitionFrameRateEnabled').value = True
            fps_enabled = True
        except Exception:
            pass
    # Set target frame rate
    try:
        nodemap.get_node('AcquisitionFrameRate').value = 30.0
    except Exception:
        try:
            nodemap.get_node('AcquisitionFrameRateAbs').value = 30.0
        except Exception:
            pass
    tl_stream_nodemap = device_triton.tl_stream_nodemap
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True
    # Prefer newest frame to reduce blocking latency
    try:
        tl_stream_nodemap['StreamBufferHandlingMode'].value = 'NewestOnly'
    except Exception:
        pass
    # Use small buffer count to reduce latency
    try:
        tl_stream_nodemap['StreamBufferCountMode'].value = 'Manual'
        tl_stream_nodemap['StreamBufferCount'].value = 4
    except Exception:
        pass

    # Print a concise summary
    try:
        resulting_fps = nodemap.get_node('AcquisitionResultingFrameRate').value
    except Exception:
        resulting_fps = None
    print(f"{TAB1}Triton configured: Bayer/Reduced res | Resulting FPS: {resulting_fps if resulting_fps else 'unknown'}")


def setup_helios_for_stream(device_helios2):
    tl_stream_nodemap = device_helios2.tl_stream_nodemap
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True
    nodemap = device_helios2.nodemap
    nodemap.get_node('PixelFormat').value = PixelFormat.Coord3D_ABCY16
    try:
        nodemap.get_node('AcquisitionMode').value = 'Continuous'
    except Exception:
        pass
    try:
        nodemap.get_node('TriggerMode').value = 'Off'
    except Exception:
        pass
    try:
        tl_stream_nodemap['StreamBufferHandlingMode'].value = 'NewestOnly'
    except Exception:
        pass


def convert_buffer_to_Coord3D_ABCY16(buffer):
    if buffer.pixel_format == enums.PixelFormat.Coord3D_ABCY16:
        return buffer
    return BufferFactory.convert(buffer, enums.PixelFormat.Coord3D_ABCY16)


'''
Frame acquisition (streaming)
'''
def acquire_triton_rgb_from_buffer(buffer):
    # Zero-copy view into buffer memory
    bytes_per_pixel = max(1, int(buffer.bits_per_pixel // 8))
    ptr = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_ubyte))
    arr = np.ctypeslib.as_array(ptr, shape=(buffer.height * buffer.width * bytes_per_pixel,))
    rgb_image = arr.reshape(buffer.height, buffer.width, bytes_per_pixel)
    return rgb_image


def acquire_helios_xyz_intensity_from_buffer(buffer, xyz_scale_mm, x_offset_mm, y_offset_mm, z_offset_mm):
    buffer_c = convert_buffer_to_Coord3D_ABCY16(buffer)
    height = int(buffer_c.height)
    width = int(buffer_c.width)
    channels_per_pixel = int(buffer_c.bits_per_pixel / 16)  # expect 4

    # Build a NumPy view over the raw 16-bit data: shape (H*W, 4) -> (A,B,C,Y)
    ptr = ctypes.cast(buffer_c.pdata, ctypes.POINTER(ctypes.c_uint16))
    total_vals = height * width * channels_per_pixel
    arr = np.ctypeslib.as_array(ptr, shape=(total_vals,))
    pixels = arr.reshape(height * width, channels_per_pixel)

    A = pixels[:, 0].astype(np.float32).reshape(height, width)
    B = pixels[:, 1].astype(np.float32).reshape(height, width)
    C = pixels[:, 2].astype(np.float32).reshape(height, width)
    Y = pixels[:, 3].reshape(height, width).astype(np.uint16)

    # Vectorized conversion to mm
    X_mm = A * xyz_scale_mm + x_offset_mm
    Y_mm = B * xyz_scale_mm + y_offset_mm
    Z_mm = C * xyz_scale_mm + z_offset_mm
    xyz_mm = np.stack((X_mm, Y_mm, Z_mm), axis=2)

    return Y, xyz_mm


'''
Live overlay
'''
def live_overlay_synced(device_triton, device_helios2):
    # Load orientation
    print(f'{TAB1}Read orientation from {ORIENTATION_FILE}')
    fs = cv2.FileStorage(ORIENTATION_FILE, cv2.FileStorage_READ)
    camera_matrix = fs.getNode('cameraMatrix').mat()
    dist_coeffs = fs.getNode('distCoeffs').mat()
    rotation_vector = fs.getNode('rotationVector').mat()
    translation_vector = fs.getNode('translationVector').mat()
    fs.release()

    # Precompute rotation matrix and cache projection components
    R, _ = cv2.Rodrigues(rotation_vector)
    t = translation_vector.reshape(3)
    K0 = camera_matrix.astype(np.float64)
    K = K0.copy()

    def project_points_numpy(xyz):
        # xyz: (N,3)
        Xc = xyz @ R.T + t
        x = Xc[:, 0] / Xc[:, 2]
        y = Xc[:, 1] / Xc[:, 2]
        # Radial + tangential distortion (k1,k2,p1,p2,k3)
        k1 = float(dist_coeffs[0, 0]) if dist_coeffs.size > 0 else 0.0
        k2 = float(dist_coeffs[1, 0]) if dist_coeffs.size > 1 else 0.0
        p1 = float(dist_coeffs[2, 0]) if dist_coeffs.size > 2 else 0.0
        p2 = float(dist_coeffs[3, 0]) if dist_coeffs.size > 3 else 0.0
        k3 = float(dist_coeffs[4, 0]) if dist_coeffs.size > 4 else 0.0
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
        x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        u = K[0, 0] * x_dist + K[0, 2]
        v = K[1, 1] * y_dist + K[1, 2]
        return np.stack((u, v), axis=1)

    # Save original pixel formats to restore later
    nodemap_triton = device_triton.nodemap
    nodemap_helios2 = device_helios2.nodemap
    pixel_format_initial_triton = nodemap_triton.get_node('PixelFormat').value
    pixel_format_initial_helios2 = nodemap_helios2.get_node('PixelFormat').value

    # Enable PTP on both devices and then configure streams
    print(f'{TAB1}Configure streams and start both cameras')
    triton_ptp = enable_ptp(device_triton, 'Triton')
    helios_ptp = enable_ptp(device_helios2, 'Helios2')
    if not (triton_ptp and helios_ptp):
        raise Exception(f'{TAB1}Failed to enable PTP on both devices. Please verify network/PTP settings.')
    setup_triton_for_stream(device_triton)
    setup_helios_for_stream(device_helios2)
    device_triton.start_stream()
    device_helios2.start_stream()

    # Cache Helios scale and offsets once (avoid nodemap reads per frame)
    nodemap = device_helios2.nodemap
    xyz_scale_mm = nodemap["Scan3dCoordinateScale"].value
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateA"
    x_offset_mm = nodemap["Scan3dCoordinateOffset"].value
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateB"
    y_offset_mm = nodemap["Scan3dCoordinateOffset"].value
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
    z_offset_mm = nodemap["Scan3dCoordinateOffset"].value

    decimation = 1  # start with full resolution; press +/- to adjust at runtime
    window_name = 'RGB with Helios overlay (synced)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Projection matrix scaling for binned RGB
    k_scaled_applied = False

    # Per-camera FPS tracking
    hlt_last_time = None
    tri_last_time = None
    hlt_fps = 0.0
    tri_fps = 0.0

    try:
        while True:
            t_loop_start = time.perf_counter()

            # Acquire buffers (near-simultaneous)
            t0 = time.perf_counter()
            buffer_h = device_helios2.get_buffer()
            t1 = time.perf_counter()
            # Update Helios FPS
            if hlt_last_time is not None:
                hlt_dt = t1 - hlt_last_time
                if hlt_dt > 0:
                    hlt_fps = 1.0 / hlt_dt
            hlt_last_time = t1

            buffer_t = device_triton.get_buffer()
            t2 = time.perf_counter()
            # Update Triton FPS
            if tri_last_time is not None:
                tri_dt = t2 - tri_last_time
                if tri_dt > 0:
                    tri_fps = 1.0 / tri_dt
            tri_last_time = t2

            # Extract frames
            t3 = time.perf_counter()
            rgb_image = acquire_triton_rgb_from_buffer(buffer_t)
            t4 = time.perf_counter()
            # Demosaic if Bayer
            if rgb_image.ndim == 3 and rgb_image.shape[2] == 1:
                rgb_image = cv2.cvtColor(np.squeeze(rgb_image, axis=2), cv2.COLOR_BayerRG2BGR)
            elif rgb_image.ndim == 2:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BayerRG2BGR)
            intensity_image, xyz_mm = acquire_helios_xyz_intensity_from_buffer(
                buffer_h, xyz_scale_mm, x_offset_mm, y_offset_mm, z_offset_mm
            )
            t5 = time.perf_counter()

            height, width = intensity_image.shape

            # Scale camera matrix once based on current RGB size vs calibration center-derived size
            if not k_scaled_applied:
                rgb_h, rgb_w = rgb_image.shape[:2]
                # Derive a baseline calibration size from principal point (assumes near-centered cx, cy)
                base_w = max(1.0, 2.0 * float(K0[0, 2]))
                base_h = max(1.0, 2.0 * float(K0[1, 2]))
                sx = float(rgb_w) / base_w
                sy = float(rgb_h) / base_h
                if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
                    K = K0.copy()
                    K[0, 0] *= sx
                    K[1, 1] *= sy
                    # Half-pixel principal point rule for binning/decimation/resizing
                    K[0, 2] = sx * (K0[0, 2] + 0.5) - 0.5
                    K[1, 2] = sy * (K0[1, 2] + 0.5) - 0.5
                k_scaled_applied = True

            # Project Helios points into Triton frame (vectorized)
            xyz_dec = xyz_mm[::decimation, ::decimation, :]
            xyz_points = xyz_dec.reshape(-1, 3)
            
            # Debug: check if we have valid XYZ data
            valid_xyz = np.isfinite(xyz_points[:, 2]) & (xyz_points[:, 2] > 0)
            num_valid_xyz = np.sum(valid_xyz)
            
            t6 = time.perf_counter()
            dist_vec = dist_coeffs.reshape(-1)
            projected_cv, _ = cv2.projectPoints(
                xyz_points.astype(np.float32),
                rotation_vector,
                translation_vector,
                K,
                dist_vec,
            )
            projected = projected_cv.reshape(-1, 2)
            t7 = time.perf_counter()

            
            # Build depth image (use Z in mm)
            depth_vals = xyz_points[:, 2]
            cols = np.round(projected[:, 0]).astype(np.int32)
            rows = np.round(projected[:, 1]).astype(np.int32)
            in_bounds = (rows >= 0) & (cols >= 0) & (rows < rgb_image.shape[0]) & (cols < rgb_image.shape[1])
            num_in_bounds = np.sum(in_bounds)
            rows = rows[in_bounds]
            cols = cols[in_bounds]
            depth_vals = depth_vals[in_bounds]

            # Z-buffer style merge: keep nearest depth per pixel
            depth_img = np.full((rgb_image.shape[0], rgb_image.shape[1]), np.inf, dtype=np.float32)
            if rows.size:
                lin_idx = rows * depth_img.shape[1] + cols
                # Reduce by min over duplicate indices
                np.minimum.at(depth_img.ravel(), lin_idx, depth_vals)
                t8 = time.perf_counter()
                # Normalize valid depths for colormap using fixed scale
                valid_mask = np.isfinite(depth_img)
                if np.any(valid_mask):
                    dmin = float(DEPTH_COLOR_MIN_MM)
                    dmax = float(DEPTH_COLOR_MAX_MM)
                    # Avoid divide-by-zero; expand range slightly if equal
                    if dmax <= dmin:
                        dmax = dmin + 1.0
                    depth_norm = np.zeros_like(depth_img, dtype=np.float32)
                    # Clip to fixed range before normalization
                    depth_clipped = np.clip(depth_img, dmin, dmax)
                    depth_norm[valid_mask] = ((depth_clipped[valid_mask] - dmin) / (dmax - dmin) * 255.0).astype(np.float32)

                    # Smoothly fill sparse regions to avoid dotty appearance (normalized convolution)
                    m = np.zeros_like(depth_norm, dtype=np.float32)
                    m[valid_mask] = 1.0
                    ksize = 9
                    g1 = cv2.getGaussianKernel(ksize, 2)
                    g2d = (g1 @ g1.T)  # sums to 1
                    num = cv2.filter2D(depth_norm, -1, g2d, borderType=cv2.BORDER_REPLICATE)
                    den = cv2.filter2D(m, -1, g2d, borderType=cv2.BORDER_REPLICATE)
                    filled = num / (den + 1e-6)
                    filled_u8 = np.clip(filled, 0, 255).astype(np.uint8)
                    mask_dense = den > 0.01

                    # Colorize and overlay only where we have support
                    heatmap = cv2.applyColorMap(filled_u8, cv2.COLORMAP_JET)
                    alpha = 0.5
                    overlay = rgb_image.copy()
                    blended = cv2.addWeighted(rgb_image, 1.0 - alpha, heatmap, alpha, 0)
                    overlay[mask_dense] = blended[mask_dense]

                    # Add color scale (JET) with min/max (mm)
                    bar_w = 20
                    margin = 10
                    bar_h = min(overlay.shape[0] - 2 * margin, 256)
                    if bar_h > 20:
                        # Build vertical gradient (top = max)
                        grad = np.linspace(255, 0, 256).astype(np.uint8).reshape(256, 1)
                        grad = np.repeat(grad, bar_w, axis=1)
                        colorbar = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
                        if bar_h != 256:
                            colorbar = cv2.resize(colorbar, (bar_w, bar_h), interpolation=cv2.INTER_NEAREST)

                        # Paste colorbar
                        y0 = margin
                        x0 = margin
                        y1 = y0 + bar_h
                        x1 = x0 + bar_w
                        overlay[y0:y1, x0:x1] = colorbar

                        # Draw border
                        cv2.rectangle(overlay, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (255, 255, 255), 1)

                        # Labels
                        top_text = f"{dmax:.0f} mm"
                        bot_text = f"{dmin:.0f} mm"
                        # Place to the right of the bar
                        tx = x1 + 8
                        cv2.putText(overlay, top_text, (tx, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(overlay, bot_text, (tx, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    t9 = time.perf_counter()
                else:
                    overlay = rgb_image
            else:
                overlay = rgb_image

            # Get center region depth for verification (average 40x40 window for better coverage)
            center_y = overlay.shape[0] // 2
            center_x = overlay.shape[1] // 2
            window_half = 20  # Larger window for sparse projection
            y0 = max(0, center_y - window_half)
            y1 = min(overlay.shape[0], center_y + window_half)
            x0 = max(0, center_x - window_half)
            x1 = min(overlay.shape[1], center_x + window_half)
            center_roi = depth_img[y0:y1, x0:x1] if rows.size else np.full((1, 1), np.inf)
            valid_center = center_roi[np.isfinite(center_roi)]
            center_depth = np.median(valid_center) if valid_center.size > 0 else np.inf
            center_depth_str = f"{center_depth:.1f} mm" if np.isfinite(center_depth) else "N/A"
            
            # Calculate coverage stats
            total_valid_pixels = np.sum(np.isfinite(depth_img)) if rows.size else 0
            total_pixels = depth_img.size
            coverage_pct = (total_valid_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

            # FPS and stats overlay
            t_show_start = time.perf_counter()
            dt = max(time.perf_counter() - t0, 1e-6)
            fps_txt = f"HLT: {hlt_fps:.1f} FPS  |  RGB: {tri_fps:.1f} FPS  |  decim={decimation}"
            cv2.putText(overlay, fps_txt, (10, overlay.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            
            # Center depth and range stats
            if rows.size and np.any(valid_mask):
                stats_txt = f"Center: {center_depth_str} | Range: {dmin:.0f}-{dmax:.0f} mm | Coverage: {coverage_pct:.1f}%"
                cv2.putText(overlay, stats_txt, (10, overlay.shape[0]-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            elif rows.size:
                stats_txt = f"Center: {center_depth_str} | Coverage: {coverage_pct:.1f}% | Increase coverage: press '+'"
                cv2.putText(overlay, stats_txt, (10, overlay.shape[0]-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
            
            # Draw crosshair at center
            cv2.line(overlay, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 255), 1)
            cv2.line(overlay, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 255), 1)

            cv2.imshow(window_name, overlay)
            t_show_end = time.perf_counter()

            # Requeue buffers
            device_helios2.requeue_buffer(buffer_h)
            device_triton.requeue_buffer(buffer_t)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key in (ord('+'), ord('=')):
                decimation = max(1, decimation - 1)
            elif key in (ord('-'), ord('_')):
                decimation = min(8, decimation + 1)

            # Adaptive decimation: prioritize coverage over raw FPS if coverage is too sparse
            fps = 1.0 / dt
            if coverage_pct < 10.0 and decimation > 1:
                # Very sparse coverage, reduce decimation even if FPS is low
                decimation = max(1, decimation - 1)
            elif fps < 25.0 and decimation < 8:
                decimation += 1
            elif fps > 32.0 and decimation > 1 and coverage_pct > 30.0:
                decimation -= 1

            # Diagnostics removed
    finally:
        # Stop streams and restore nodes
        device_triton.stop_stream()
        device_helios2.stop_stream()
        nodemap_triton.get_node('PixelFormat').value = pixel_format_initial_triton
        nodemap_helios2.get_node('PixelFormat').value = pixel_format_initial_helios2
        cv2.destroyAllWindows()


def example_entry_point():
    print(f'{TAB1}py_HLTRGB_3_Overlay (Synced Live)')

    if not exists(ORIENTATION_FILE):
        print(f"{TAB1}File '{ORIENTATION_FILE}' not found. Please run 'py_HLTRGB_1_calibration' and 'py_HLTRGB_2_orientation' first.")
        return

    devices = create_devices_with_tries()
    applicable_triton = get_applicable_devices(devices, TRITON)
    applicable_helios = get_applicable_devices(devices, HELIOS2)

    device_triton = system.select_device(applicable_triton)
    device_helios2 = system.select_device(applicable_helios)

    live_overlay_synced(device_triton, device_helios2)

    system.destroy_device()
    print(f'{TAB1}Destroyed all created devices')


if __name__ == '__main__':
    print('Example started\n')
    example_entry_point()
    print('\nExample finished')


