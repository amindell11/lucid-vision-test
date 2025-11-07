# Lucid Vision Camera Calibration & Overlay

Python scripts for calibrating and overlaying 3D depth data from Lucid Vision Helios2 cameras onto Triton RGB camera images.

## Overview

This project contains a 3-part pipeline for combining 3D depth sensing with RGB imaging:

1. **py_HLTRGB_1_calibration.py** - Calibrate the Triton RGB camera lens (intrinsics and distortion)
2. **py_HLTRGB_2_orientation.py** - Calculate the orientation between Helios2 and Triton cameras
3. **py_HLTRGB_3_overlay_synced.py** - Live overlay of Helios2 depth data on Triton RGB stream

## Hardware Requirements

- **Lucid Vision Triton RGB Camera** (TRI series with -C suffix, e.g., TRI050S-C)
- **Lucid Vision Helios2 3D Camera** (HLT/HTP/HTW series)
- Calibration target with circular pattern (5x4 white circles) - [Download PDF](https://arenasdk.s3-us-west-2.amazonaws.com/LUCID_target_whiteCircles.pdf)

## Software Requirements

### 1. Install Lucid Vision ArenaSDK

**ArenaSDK is required** and must be installed before the Python packages.

#### Windows:
1. Download ArenaSDK from [Lucid Vision Labs Support](https://thinklucid.com/downloads-hub/)
2. Run the installer (e.g., `ArenaSDK_v0.x.xx_Windows_x64.exe`)
3. The installer places Python bindings in the SDK directory (typically `C:\Program Files\Lucid Vision Labs\Arena SDK\ArenaSDK_v0.x.xx\Python`)

**To use arena_api in your conda/venv (REQUIRED):**

Run the automated setup script (recommended):
```bash
python setup_arena_env.py
```

This script automatically:
- Finds your ArenaSDK installation
- Detects your conda/venv environment
- Creates a `.pth` file to link arena_api
- Verifies the setup works

**Manual setup (if needed):**
```bash
# Create arena_sdk.pth with path to ArenaSDK Python folder
echo C:\Program Files\Lucid Vision Labs\Arena SDK\ArenaSDK_v0.x.xx\Python > %CONDA_PREFIX%\Lib\site-packages\arena_sdk.pth

# Verify
python -c "import arena_api; print('arena_api:', arena_api.__version__)"
```

#### Linux:
1. Download ArenaSDK for Linux
2. Extract and run the installer script
3. Source the arena setup script:
   ```bash
   source /opt/ArenaSDK_Linux_x64/arena_api-x.x.x-py3-none-any.whl
   ```

### 2. Install Python Dependencies

After ArenaSDK is installed:

```bash
# Create a virtual environment (recommended)
python -m venv venv
# Or with conda:
# conda create -n lucid-vis python=3.9

# Activate the environment
# Windows:
venv\Scripts\activate
# Or conda:
# conda activate lucid-vis

# Install Python packages
pip install -r requirements_win.txt
```

## Usage

### Step 1: Calibrate Triton Camera

```bash
python py_HLTRGB_1_calibration.py
```

- Place the calibration target at your working distance
- Focus the Triton lens on the target
- Move the target to 10 different positions in the field of view
- The script captures images automatically and saves calibration to `tritoncalibration.yml`

### Step 2: Calculate Camera Orientation

```bash
python py_HLTRGB_2_orientation.py
```

- Place the calibration target in the center of both cameras' fields of view
- Keep the target at your application's working distance
- **Do not move the target or cameras** during capture
- The script saves orientation data to `orientation.yml`

**Important:** For best results, make the calibration target 3D by:
- Placing it on a textured surface (cardboard, foam)
- Sticking small objects (foam dots, coins) on the circles
- The Helios2 needs measurable depth, not just a flat surface

### Step 3: Live Overlay

```bash
python py_HLTRGB_3_overlay_synced.py
```

- Real-time depth overlay on RGB video
- Press `+`/`-` to adjust point cloud decimation
- Press `q` or `ESC` to quit
- Shows depth colormap, FPS, and center depth measurement

## Output Files

- `tritoncalibration.yml` - Triton camera intrinsics and distortion coefficients
- `orientation.yml` - Relative orientation (rotation & translation) between cameras

## Troubleshooting

### "No module named 'arena_api'"
- ArenaSDK must be installed first (see Software Requirements above)
- Verify `arena_api` is in your Python path

### "Unable to find points in TRI/HLT image"
- Ensure calibration target is in focus and well-lit
- Check that all 20 circles are visible in frame
- Try adjusting camera exposure settings

### "Calibration circles have NO VALID DEPTH"
- The Helios2 cannot see depth on flat white surfaces
- Make the target 3D (see Step 2 notes above)
- Ensure proper working distance (typically 0.3-2m for Helios2)

### Version Compatibility Issues
- This code is tested with Python 3.8-3.10
- Use numpy 1.21+, opencv-python 4.8+
- Older versions in original requirements caused build errors

## License

Copyright (c) 2024, Lucid Vision Labs, Inc.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

## Support

For issues with:
- **ArenaSDK installation**: [Lucid Vision Support](https://support.thinklucid.com/)
- **This code**: Open an issue on this repository

