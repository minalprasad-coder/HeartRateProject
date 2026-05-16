import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt

# Store green signal values
green_signal = []

# Open webcam
camera = cv2.VideoCapture(0)

# MediaPipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# Signal filtering function
def filter_signal(signal):

    low = 0.7
    high = 4.0
    fs = 30  # webcam FPS

    nyquist = fs / 2

    low = low / nyquist
    high = high / nyquist

    b, a = butter(3, [low, high], btype='band')

    filtered = filtfilt(b, a, signal)

    return filtered


# BPM calculation
def calculate_bpm(signal):

    fft = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(len(signal), d=1/30)

    fft_magnitude = np.abs(fft)

    # Ignore unrealistic frequencies
    valid_idx = np.where((frequencies >= 0.7) &
                         (frequencies <= 4.0))

    valid_freqs = frequencies[valid_idx]
    valid_fft = fft_magnitude[valid_idx]

    if len(valid_fft) == 0:
        return 0

    strongest_frequency = valid_freqs[
        np.argmax(valid_fft)
    ]

    bpm = strongest_frequency * 60

    return int(bpm)


while True:

    success, frame = camera.read()

    if not success:
        break

    # Flip camera (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
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

            # Forehead ROI
            forehead = frame[
                y:y + 50,
                x:x + width
            ]

            # Draw forehead box
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + 50),
                (255, 0, 0),
                2
            )

            # Avoid empty ROI
            if forehead.size > 0:

                # Mean green intensity
                green_value = np.mean(
                    forehead[:, :, 1]
                )

                green_signal.append(green_value)

                # Keep only latest 300 samples
                if len(green_signal) > 300:
                    green_signal = green_signal[-300:]

                    try:
                        filtered_signal = filter_signal(
                            green_signal
                        )

                        bpm = calculate_bpm(
                            filtered_signal
                        )

                        cv2.putText(
                            frame,
                            f'Heart Rate: {bpm} BPM',
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                    except:
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