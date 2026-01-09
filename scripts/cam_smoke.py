import time
import cv2
import mediapipe as mp
from pathlib import Path

# --- resolve project root (where .git lives), so the notebook can run from anywhere ---
root = Path.cwd()
while root != root.parent and not (root / ".git").exists():
    root = root.parent

MODEL_PATH = root / "models" / "hand_landmarker.task"
print("Resolved MODEL_PATH:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Скачай модель в корень репозитория:\n"
        "mkdir -p models\n"
        "curl -L -o models/hand_landmarker.task "
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

WIN = "MediaPipe HandLandmarker (q/ESC to quit)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Камера не открылась. Попробуй VideoCapture(1).")

start = time.time()

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Не удалось прочитать кадр с камеры")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                h, w = frame.shape[:2]
                for hand in result.hand_landmarks:
                    for lm in hand:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.imshow(WIN, frame)

            # если окно закрыли крестиком — выходим до следующего imshow
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test done")
