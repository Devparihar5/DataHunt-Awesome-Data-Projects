import datetime
from ultralytics import YOLO
import cv2 as cv
from helpe import create_video_writer


CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)

capcture = cv.VideoCapture(0)
writer = create_video_writer(capcture, "Media")

model = YOLO("yolov8n.pt")

while True:
    start = datetime.datetime.now()

    isTrue, frame = capcture.read()

    detections = model(frame)[0]

    for data in detections.boxes.data.tolist():
        confidence = data[4]

        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        x, y, w, h = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv.rectangle(frame, (x, y), (x + w, y + h), GREEN, thickness=2)

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
