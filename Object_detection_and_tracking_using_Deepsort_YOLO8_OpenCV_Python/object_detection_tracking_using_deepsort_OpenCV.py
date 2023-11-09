import datetime
from ultralytics import YOLO
import cv2 as cv
from helpe import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
capcture = cv.VideoCapture(0)
writer = create_video_writer(capcture, "Media")

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=50)
while True:
    start = datetime.datetime.now()

    isTrue, frame = capcture.read()

    detections = model(frame)[0]
    results = []

    for data in detections.boxes.data.tolist():
        confidence = data[4]

        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        x, y, w, h = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        results.append([[x, y, x + w, y + h], confidence, class_id])

    tracks = tracker.update_tracks(results, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        x, y, w, h = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        cv.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        cv.rectangle(frame, (x, y - 20), (x + 20, y), GREEN, -1)
        cv.putText(
            frame, str(track_id), (x + 5, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2
        )

    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total*1000:.0f} milliseconds")

    fps = f"FPS: {1/total:.2f}"
    cv.putText(frame, fps, (50, 50), cv.FONT_HERSHEY_COMPLEX, 2, GREEN, thickness=2)

    cv.imshow("Mafia", frame)
    writer.write(frame)
    if cv.waitKey(20) & 0xFF == ord("d"):
        break


capcture.release()
writer.release()
cv.destroyAllWindows()
