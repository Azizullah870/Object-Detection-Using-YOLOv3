import cv2
import numpy as np

# Set parameters
wht = 320
threshold = 0.5

# Open video file
cap = cv2.VideoCapture("F:\Video1.mp4")

# Load class names
classesFile = "coco.names"
with open(classesFile, "rt") as f:
    classesNames = f.read().rstrip('\n').split('\n')

# Load YOLO model
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv2.dnn.readNet(modelConfiguration, modelWeights)

# Use OpenCV as backend and target
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findobjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    ClassIds = []
    cnfs = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # YOLOv3 has the 5th position for confidence, rest are class scores
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                ClassIds.append(classId)
                cnfs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, cnfs, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = f'{classesNames[ClassIds[i]].upper()} {int(cnfs[i] * 100)}%'
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # To give the image input in blob format
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (wht, wht), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Extract output layers
    layerNames = net.getLayerNames()
    outputNames = net.getUnconnectedOutLayersNames()
    outputs = net.forward(outputNames)

    # Find objects in the frame
    findobjects(outputs, frame)
    frame=cv2.resize(frame, (1000, 800))
    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
