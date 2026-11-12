# RL Test: Camera-Controlled MuJoCo Robot

This project connects a webcam to a MuJoCo simulation:

1. OpenCV captures frames from a V4L2 camera.
2. MediaPipe Pose Landmarker detects body landmarks.
3. The cosine of the angle between the right upper arm and forearm is calculated.
4. That value is written to MuJoCo control slot `data.ctrl[2]` for the
   `robotstudio_so101` model.
5. The camera feed, pose landmarks, and calculated cosine are displayed in an
   OpenCV window while the MuJoCo viewer runs.

The repository is currently a camera-to-simulation control prototype; it does
not include reinforcement-learning training code.

## Why this project exists

This project is being developed to learn MuJoCo, robotics simulation, and the
practical steps involved in connecting real-world camera data to a simulated
robot.

## Project milestones

1. **Get MuJoCo working with OpenCV**  
   Capture webcam input, process it with OpenCV and MediaPipe, and use the
   resulting pose information to control a MuJoCo simulation.
2. **Collect data using this project**  
   Record useful camera, pose, and simulation data that can support later
   experiments and model training.
3. **Learn to use foundation models and VLA models**  
   Explore how vision-language and vision-language-action models can be used
   for robotics tasks and simulated robot control.

## Requirements

- Linux, or Windows with a working WSL camera passthrough
- Python 3.14 or newer
- A webcam available as `/dev/video0` (or another V4L2 device)
- [`uv`](https://docs.astral.sh/uv/)
- A graphical session for the MuJoCo viewer and OpenCV window

The project uses:

- `mujoco-menagerie` for the `robotstudio_so101` model
- `mediapipe` for pose detection
- OpenCV for camera capture and display
- NumPy for vector and cosine calculations

## Setup

Install the locked dependencies from the repository root:

```bash
uv sync
```

The MediaPipe model asset is expected at the repository root. This project
currently uses:

```text
pose_landmarker_heavy.task
```

`pose_landmarker_full.task` is also included for experimentation, but is not
selected by default.

## Camera setup on WSL

If the webcam is connected to Windows and the program runs inside WSL, first
attach it to WSL. Review `attach_camera.sh` and replace `1-3` with the correct
USB bus ID for your camera:

```bash
chmod +x attach_camera.sh
./attach_camera.sh
```

Check that the camera is visible and identify its device index:

```bash
ls /dev/video*
fuser -v /dev/video*
```

To verify that the camera produces video before running the application:

```bash
ffplay /dev/video0
```

The capture settings are currently hard-coded in `main.py` to camera index
`0`, MJPEG, and a resolution of `854x480`. Change the `VideoCapture` index or
settings in `camera_and_pose_thread()` if your camera uses a different device.

## Run

From the repository root:

```bash
uv run python main.py
```

Two windows should appear:

- MuJoCo's passive viewer with the SO101 model
- An OpenCV window titled `Camera Feed + MediaPipe`

Press `q` in the OpenCV window to stop the camera thread, close the windows,
and exit cleanly.

## Project layout

```text
main.py                    Camera, pose detection, and MuJoCo control loop
attach_camera.sh           WSL USB camera attach and diagnostic commands
pose_landmarker_heavy.task MediaPipe pose model used at runtime
pose_landmarker_full.task  Alternate MediaPipe pose model
pyproject.toml             Project metadata and dependencies
uv.lock                    Locked dependency versions
mujoco-3.15.0/             Local MuJoCo distribution and examples
```

## Implementation notes

- Camera capture and MediaPipe inference run in a background thread.
- MuJoCo stepping and OpenCV GUI calls run on the main thread.
- Shared camera data is protected by a threading lock.
- Pose landmarks must pass visibility and presence checks for the wrist before
  the MuJoCo control value is updated.
- The current control mapping uses the right shoulder, elbow, and wrist
  landmarks and writes the resulting cosine to `data.ctrl[2]`.

## Troubleshooting

### Camera cannot be opened

Confirm that the device exists and is not already in use:

```bash
ls -l /dev/video*
fuser -v /dev/video*
```

On WSL, rerun the USB attach commands with the correct bus ID. If the camera is
not `/dev/video0`, update the index in `main.py`.

### MediaPipe model file not found

Run the application from the repository root and verify that
`pose_landmarker_heavy.task` exists there.

### MuJoCo viewer does not open

Run from a graphical Linux session with an available display. In WSL, ensure
your WSLg/X server configuration supports OpenGL applications.

### OpenCV window does not receive `q`

Click the `Camera Feed + MediaPipe` window before pressing `q`. OpenCV keyboard
events are processed by the window that has focus.