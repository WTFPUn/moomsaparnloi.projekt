import cv2
import numpy as np

def cornerInshape(box, motorbikePOS) :
  corner = {
    "upLeft" : (box[0], box[1]),
    "upRight" : (box[0] + box[2], box[1]),
    "downLeft" : (box[0], box[1] + box[3]),
    "downRight" : (box[0] + box[2], box[1] + box[3])
  }
  checkpos = []
  for i in corner:
      checkpos.append(True if all(corner[i][0] in range(motorbikePOS[0], motorbikePOS[0] + motorbikePOS[2]), corner[i][1] in range(motorbikePOS[1], motorbikePOS[1] + motorbikePOS[3])) else False)
  return any(checkpos)

with open('piford.names', 'r') as f:
  classes = f.read().splitlines()

  net = cv2.dnn.readNetFromDarknet('yolov4-custom.cfg', 'yolov4-custom_best.weights')
  model = cv2.dnn_DetectionModel(net)
  model.setInputParams(scale = 1 / 255, size = (416, 416), swapRB = True)

img = cv2.resize(cv2.imread("imgout/47.jpg"), (1280, 720))
cv2.line(img, (0, 432), (1280, 432), color = (0, 0, 255), thickness = 2)
classIds, scores, boxes = model.detect(img, confThreshold = 0.6, nmsThreshold = 0.4)
print(classIds)
print(scores)
print(boxes)
check = []
for(classId, score, box) in zip(classIds, scores, boxes):
  if classId == 1 and box[1] in range(429,435) :
    check = []
    for i in range(len(classIds)):
      if classIds[i] == 0:
        check.append(True if cornerInshape(box, boxes[i]) else False)
  cv2.rectangle(img, (box[0], box[1],), (box[0] + box[2], box[1] + box[3]), color = (0, 255, 0), thickness = 2)
  text = "%s: %.2f" % (classes[classId[0]], score)
  cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color = (0, 255, 0), thickness = 1)

print(any(check))
cv2.imshow('Image', img)
cv2.waitKey(0)