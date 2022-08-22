import cv2, time, torch
from pytorchyolo import detect, models

# Load the YOLO model
model = models.load_model(
#   "config\\yolov3.cfg", 
#   "weights\\yolov3.weights")
  "config\\yolov3-tiny.cfg", 
  "weights\\yolov3-tiny.weights")

def inference(n):
    for _ in range(n):
        # Load the image as a numpy array
        img = cv2.imread("H:\\workspace\\luotongan\\CamData-Datasets\\CamData2\\person\\person-8\\img\\1658726163946.jpg")

        # Convert OpenCV bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Runs the YOLO model on the image 
        boxes = detect.detect_image(model, img)

        # print(boxes)
        # Output will be a numpy array in the following format:
        # [[x1, y1, x2, y2, confidence, class]]


with torch.no_grad():
  # warmup
  inference(20)

  # speedtest
  tic = time.time()
  inference(100)
  # inference(1)
  toc = time.time()

print(100/(toc - tic))