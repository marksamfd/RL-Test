import threading
from time import time
import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import mujoco_menagerie as mm
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

# Initialize MuJoCo model
robot_name = "robotstudio_so101"
model = mm.load(robot_name)
data = mujoco.MjData(model)

# Shared thread variables
latest_frame = None
latest_landmarks = None
frame_lock = threading.Lock()
stop_event = threading.Event()

# Initialize ediaPipe Pose


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def cos_angle_between_vectors(line1, line2):
    # Convert inputs to numpy arrays
    x1l1, y1l1, x2l1, yl1 = line1
    x1l2, y1l2, x2l2, yl2 = line2
    u = np.array([x2l1 - x1l1, yl1 - y1l1])
    v = np.array([x2l2 - x1l2, yl2 - y1l2])

    # Calculate dot product and magnitudes
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Avoid division by zero
    if norm_u == 0 or norm_v == 0:
        raise ValueError("Cannot calculate angle with a zero vector.")

    # Clip the cosine value to avoid floating-point errors outside [-1.0, 1.0]
    cosine_theta = np.clip(dot_product / (norm_u * norm_v), -1.0, 1.0)

    return cosine_theta


def calculate_cos_arm_forearm(pose_landmarks_list):
    line1 = (
        pose_landmarks_list[0][12].x,
        pose_landmarks_list[0][12].y,
        pose_landmarks_list[0][14].x,
        pose_landmarks_list[0][14].y,
    )
    line2 = (
        pose_landmarks_list[0][16].x,
        pose_landmarks_list[0][16].y,
        pose_landmarks_list[0][14].x,
        pose_landmarks_list[0][14].y,
    )
    cosine_angle = cos_angle_between_vectors(line1, line2)
    return cosine_angle


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    # annotated_image = np.zeros_like(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )

    cv2.putText(
        annotated_image,
        "Press 'q' to exit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cosine_angle = calculate_cos_arm_forearm(pose_landmarks_list)
    cv2.putText(
        annotated_image,
        f"Cosine angle between arm and forearm: {cosine_angle:.2f}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return annotated_image


def camera_and_pose_thread():
    global latest_frame, latest_landmarks

    # Configure V4L2 with MJPEG to prevent USB/WSL buffer corruption
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Discard startup frames
        for _ in range(5):
            cap.read()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="pose_landmarker_heavy.task",
                # delegate=python.BaseOptions.Delegate.GPU,  # GPU
            ),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.85,
        )

        timestamp_ms = 0
        with PoseLandmarker.create_from_options(options) as landmarker:
            print(f"is cap opened: {cap.isOpened()}, landmarker: {landmarker is not None}, {landmarker}")
            while not stop_event.is_set() and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    continue

                # Convert BGR to RGB for MediaPipe processing.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                rgb_frame.flags.writeable = False
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgb_frame)

                # VIDEO mode returns the result directly and avoids asynchronous
                # work still being queued when the landmarker is closed.
                timestamp_ms = max(timestamp_ms + 1, int(time() * 1000))
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                with frame_lock:
                    latest_frame = frame
                    if results.pose_landmarks:
                        latest_frame = draw_landmarks_on_image(frame, results)
                    latest_landmarks = results.pose_landmarks
    finally:
        cap.release()


def main():
    # Start background thread for VideoCapture + MediaPipe
    camera_thread = threading.Thread(target=camera_and_pose_thread, daemon=True)
    camera_thread.start()
    print("Camera thread started")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            try:
                while viewer.is_running():
                    # Step MuJoCo physics engine
                    mujoco.mj_step(model, data)

                    # Map Pose Landmarks to MuJoCo Joint Controls here if needed
                    with frame_lock:
                        landmarks = latest_landmarks
                        display_frame = (
                            latest_frame.copy() if latest_frame is not None else None
                        )
                    if landmarks is not None and len(landmarks) > 0:
                        if (
                            landmarks[0][14].visibility > 0.5
                            and landmarks[0][14].presence > 0.5
                        ):
                            # Example: Use right wrist Y-coordinate to control joint 0
                            # wrist_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
                            data.ctrl[2] = calculate_cos_arm_forearm(
                                landmarks
                            )  # Example control based on cosine angle

                    viewer.sync()

                    # OpenCV GUI execution strictly on the MAIN thread
                    if display_frame is not None:
                        cv2.imshow("Camera Feed + MediaPipe", display_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            finally:
                # Stop and join the camera thread before the viewer context exits.
                stop_event.set()
                camera_thread.join(timeout=5)
    finally:
        stop_event.set()
        camera_thread.join(timeout=5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
