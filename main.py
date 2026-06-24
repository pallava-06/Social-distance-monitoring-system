from scipy.spatial import distance as dist
import numpy as np
import cv2
from ultralytics import YOLO

INPUT_VIDEO = "video.mp4"
OUTPUT_VIDEO = "output.mp4"

MIN_CONF = 0.15
MIN_DISTANCE = 50

print("[INFO] loading YOLOv11 model...")
model = YOLO("yolo11n.pt")   # use yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.

cap = cv2.VideoCapture(INPUT_VIDEO)
writer = None

# Get the class id for person
names = model.names

person_idx = next((i for i, n in names.items() if n == "person"), 0)


def detect_people(frame, model, person_idx=0, min_conf=0.3):
    results = model(frame, imgsz=1280, verbose=False)[0]  # one Results object for this frame
    detections = []

    if results.boxes is None or len(results.boxes) == 0:
        return detections

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    for box, cls, conf in zip(boxes, classes, confs):
        if cls != person_idx or conf < min_conf:
            continue

        startX, startY, endX, endY = box.astype(int)
        cX = int((startX + endX) / 2)
        cY = int((startY + endY) / 2)

        detections.append((float(conf), (startX, startY, endX, endY), (cX, cY)))

    return detections

while True:
    grabbed, frame = cap.read()
    if not grabbed:
        break

    frame = cv2.resize(frame, (1200, int(frame.shape[0] * 1280 / frame.shape[1])))

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    results = detect_people(frame, model, person_idx=person_idx, min_conf=MIN_CONF)

    violate = set()

    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)

    for i, (prob, bbox, centroid) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 0, 255) if i in violate else (0, 255, 0)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 4, color, 1)

    text = f"Social Distancing Violations: {len(violate)}"
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    writer.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()