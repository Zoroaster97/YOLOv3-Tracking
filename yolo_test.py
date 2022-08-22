import cv2, time
from pytorchyolo import detect, models

# Load the YOLO model
model = models.load_model(
#   "config\\yolov3.cfg", 
#   "weights\\yolov3.weights")
  "config\\yolov3-tiny.cfg", 
  "weights\\yolov3-tiny.weights")


# Load the image as a numpy array
# img = cv2.imread("H:\\workspace\\luotongan\\CamData-Datasets\\CamData2\\person\\person-8\\img\\1658726163946.jpg")
img = cv2.imread("H:\\workspace\\luotongan\\SavedCamData\\test1658725823165\\cam1\\1658725881514.jpg")

# Convert OpenCV bgr to rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Runs the YOLO model on the image 
boxes = detect.detect_image(model, img)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    # if cls == 0 and conf > 0.5:
    if cls == 0:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        print(cls)
cv2.imshow('img', img)
cv2.waitKey()


# print(boxes)
# Output will be a numpy array in the following format:
# [[x1, y1, x2, y2, confidence, class]]