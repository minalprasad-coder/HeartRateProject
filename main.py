import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt

# Setup
camera = cv2.VideoCapture(0)

FPS = 30
WINDOW_SIZE = 450  # 15 sec

signal_values = []
bpm_history = []

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Filter signal
# -----------------------------
def filter_signal(signal):

    low = 0.7
    high = 4.0

    nyquist = FPS / 2

    low = low / nyquist
    high = high / nyquist

    b, a = butter(
        3,
        [low, high],
        btype='band'
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


# -----------------------------
# Calculate BPM
# -----------------------------
def calculate_bpm(signal):

    fft = np.fft.rfft(signal)

    freqs = np.fft.rfftfreq(
        len(signal),
        d=1/FPS
    )

    magnitude = np.abs(fft)

    valid_idx = np.where(
        (freqs >= 0.7) &
        (freqs <= 4.0)
    )

    valid_freqs = freqs[valid_idx]
    valid_magnitude = magnitude[valid_idx]

    if len(valid_magnitude) == 0:
        return 0

    dominant_frequency = valid_freqs[
        np.argmax(valid_magnitude)
    ]

    bpm = dominant_frequency * 60

    return int(bpm)


# -----------------------------
# Get ROI from landmarks
# -----------------------------
def get_roi(frame, landmarks, indices):

    h, w, _ = frame.shape

    points = []

    for idx in indices:
        x = int(
            landmarks[idx].x * w
        )
        y = int(
            landmarks[idx].y * h
        )

        points.append((x, y))

    points = np.array(points)

    x, y, width, height = cv2.boundingRect(points)

    roi = frame[
        y:y + height,
        x:x + width
    ]

    return roi, (x, y, width, height)


# -----------------------------
# Main loop
# -----------------------------
while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:

        face_landmarks = (
            result.multi_face_landmarks[0]
            .landmark
        )

        # ROI landmark points
        forehead_points = [
            10, 67, 103, 109, 338,
            297, 332, 284
        ]

        left_cheek_points = [
            50, 101, 118, 119,
            120, 121, 205
        ]

        right_cheek_points = [
            280, 330, 347,
            348, 349, 350,
            425
        ]

        # Extract ROIs
        forehead, f_box = get_roi(
            frame,
            face_landmarks,
            forehead_points
        )

        left_cheek, lc_box = get_roi(
            frame,
            face_landmarks,
            left_cheek_points
        )

        right_cheek, rc_box = get_roi(
            frame,
            face_landmarks,
            right_cheek_points
        )

        # Draw ROI boxes
        for box in [
            f_box,
            lc_box,
            rc_box
        ]:

            x, y, w, h = box

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

        # Check ROIs exist
        if (
            forehead.size > 0
            and left_cheek.size > 0
            and right_cheek.size > 0
        ):

            # Green signals
            forehead_green = np.mean(
                forehead[:, :, 1]
            )

            left_green = np.mean(
                left_cheek[:, :, 1]
            )

            right_green = np.mean(
                right_cheek[:, :, 1]
            )

            # Average signal
            final_signal = (
                forehead_green +
                left_green +
                right_green
            ) / 3

            signal_values.append(
                final_signal
            )

            # Keep latest values
            if len(signal_values) > WINDOW_SIZE:

                signal_values = (
                    signal_values[
                        -WINDOW_SIZE:
                    ]
                )

                try:

                    filtered_signal = (
                        filter_signal(
                            signal_values
                        )
                    )

                    # Extra smoothing
                    filtered_signal = (
                        np.convolve(
                            filtered_signal,
                            np.ones(5)/5,
                            mode='same'
                        )
                    )

                    bpm = calculate_bpm(
                        filtered_signal
                    )

                    # Ignore impossible BPM
                    if 50 <= bpm <= 120:

                        bpm_history.append(
                            bpm
                        )

                    if len(
                        bpm_history
                    ) > 10:

                        bpm_history.pop(0)

                    stable_bpm = int(
                        np.mean(
                            bpm_history
                        )
                    )

                    cv2.putText(
                        frame,
                        f'Heart Rate: '
                        f'{stable_bpm} BPM',
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                except Exception:
                    pass

    cv2.imshow(
        "Advanced rPPG Heart Rate",
        frame
    )

    if (
        cv2.waitKey(1)
        & 0xFF == ord('q')
    ):
        break

camera.release()
cv2.destroyAllWindows()
