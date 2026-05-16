import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt

# Store signal values
green_signal = []
bpm_history = []

# Webcam
camera = cv2.VideoCapture(0)

# MediaPipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

FPS = 30
WINDOW_SIZE = 450  # 15 seconds


# -----------------------------
# Signal Filtering
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
# BPM Calculation
# -----------------------------
def calculate_bpm(signal):

    fft = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1/FPS
    )

    magnitude = np.abs(fft)

    # Valid heart-rate range
    valid_idx = np.where(
        (frequencies >= 0.7) &
        (frequencies <= 4.0)
    )

    valid_freqs = frequencies[valid_idx]
    valid_mag = magnitude[valid_idx]

    if len(valid_mag) == 0:
        return 0

    dominant_frequency = valid_freqs[
        np.argmax(valid_mag)
    ]

    bpm = dominant_frequency * 60

    return int(bpm)


# -----------------------------
# Main Loop
# -----------------------------
while True:

    success, frame = camera.read()

    if not success:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # RGB conversion
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Face detection
    result = face_detector.process(rgb)

    if result.detections:

        for detection in result.detections:

            bbox = detection.location_data.relative_bounding_box

            h, w, c = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Prevent negative coordinates
            x = max(0, x)
            y = max(0, y)

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            # -----------------------------
            # Better forehead ROI
            # Use center forehead only
            # -----------------------------
            forehead = frame[
                y:y + 50,
                x + width//4:
                x + 3*width//4
            ]

            # Draw forehead rectangle
            cv2.rectangle(
                frame,
                (x + width//4, y),
                (x + 3*width//4, y + 50),
                (255, 0, 0),
                2
            )

            if forehead.size > 0:

                # Green channel mean
                green_value = np.mean(
                    forehead[:, :, 1]
                )

                green_signal.append(
                    green_value
                )

                # Keep latest samples
                if len(green_signal) > WINDOW_SIZE:
                    green_signal = green_signal[-WINDOW_SIZE:]

                    try:
                        # Filter signal
                        filtered_signal = filter_signal(
                            green_signal
                        )

                        # Calculate BPM
                        bpm = calculate_bpm(
                            filtered_signal
                        )

                        # Smooth BPM
                        bpm_history.append(bpm)

                        if len(bpm_history) > 10:
                            bpm_history.pop(0)

                        stable_bpm = int(
                            np.mean(bpm_history)
                        )

                        # Display BPM
                        cv2.putText(
                            frame,
                            f'Heart Rate: {stable_bpm} BPM',
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                    except Exception:
                        pass

    cv2.imshow(
        "Heart Rate Detection using rPPG",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
