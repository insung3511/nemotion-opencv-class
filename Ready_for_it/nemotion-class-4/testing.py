from imutils.video import FPS
import numpy as np
import imutils
import dlib
import cv2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

videoStream = "/Users/bahk_insung/Documents/GitHub/nemotion-opencv-class/Ready_for_it/nemotion-class-4/input/cat.mp4"
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(videoStream)
tracker = None
label = ""

fps = FPS().start()
while True:
    (grabbed, frame) = vs.read()
    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if tracker is None:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

        net.setInput(blob)
        detections = net.forward()

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            conf = detections[0, 0, i, 2]
            label = CLASSES[int(detections[0, 0, i, 2])]

            if conf > 0.2 and label == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    else:
        tracker.update(rgb)
        pos = tracker.get_position()

        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())

        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 15), 
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.45, (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.release()