import cv2
import datetime
import os

def cornerInshape(box, motorbikePOS):
  corner = {
    "upLeft" : (box[0], box[1]),
    "upRight" : (box[0] + box[2], box[1]),
    "downLeft" : (box[0], box[1] + box[3]),
    "downRight" : (box[0] + box[2], box[1] + box[3])
  }
  checkpos = []
  for i in corner:
      checkpos.append(True if all([corner[i][0] in range(motorbikePOS[0], motorbikePOS[0] + motorbikePOS[2]), corner[i][1] in range(motorbikePOS[1], motorbikePOS[1] + motorbikePOS[3])]) else False)
  return any(checkpos)
          
def ExportImage(img, box, date:datetime):
  imgout = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
  fulldate =  {
              "year" : date.strftime("%Y"),
              "month" : date.strftime("%b"),
              "date" : date.strftime("%d"),
              "hour" : date.strftime("%H"),
              "min" : date.strftime("%M"),
              "sec" : date.strftime("%S"),
            }
  cond = fulldate["year"] + "-" + fulldate["month"] + "-" + fulldate["date"]
  filename = fulldate["hour"] + "." + fulldate["min"] + "." + fulldate["sec"]
  if cond not in os.listdir("mBikeHeadDetect"):
    os.mkdir(f"mBikeHeadDetect/{cond}")
  cv2.imwrite(f"mBikeHeadDetect/{cond}/{filename}.jpg", imgout)



cap = cv2.VideoCapture('videotest.mp4')

with open('piford.names', 'r') as f:
  classes = f.read().splitlines()

  net = cv2.dnn.readNetFromDarknet('yolov4-custom.cfg', 'yolov4-custom_best.weights')
  model = cv2.dnn_DetectionModel(net)
  model.setInputParams(scale = 1 / 255, size = (416, 416), swapRB = True)
width = 1080
height = 960
while True:
  success, img  =  cap.read()
  img = cv2.resize(img, (width, height))
  cv2.line(img, (0, int(height*2/3)), (width, int(height*2/3)), color = (0, 0, 255), thickness = 2)
  classIds, scores, boxes = model.detect(img, confThreshold = 0.6, nmsThreshold = 0.4)
  check = [False]
  for(classId, score, box) in zip(classIds, scores, boxes):
    if classId == 1 and box[1] in range(int(height*2/3) - 3, int(height*2/3) + 3):
      check = [False]
      for i in range(len(classIds)):
        if classIds[i] == 0:
          check.append(True if cornerInshape(box, boxes[i]) else False)
      print(any(check))
    elif classId == 2 and box[1] in range(int(height*2/3) - 3, int(height*2/3) + 3):
      check = [False]
      for i in range(len(classIds)):
        if classIds[i] == 0:
          check.append(False if cornerInshape(box, boxes[i]) else True)
          if cornerInshape(box, boxes[i]):
            current = datetime.datetime.now()
            ExportImage(img, boxes[i], current)
      print(all(check))
    cv2.rectangle(img, (box[0], box[1],), (box[0] + box[2], box[1] + box[3]), color = (0, 255, 0), thickness = 2)
    text = "%s: %.2f" % (classes[classId[0]], score)
    cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color = (0, 255, 0), thickness = 2)
  cv2.imshow('Image', img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break