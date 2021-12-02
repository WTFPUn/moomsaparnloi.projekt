import os
import random
imgList = os.listdir("imgout")
random.shuffle(imgList)
print(imgList)
textFileTrain = open("train.txt","w")
textFileTest = open("test.txt", "w")
countImg = len(imgList)
for i in range(len(imgList)):
  if i < len(imgList) * .9:
    textFileTrain.write("data/obj/" + imgList[i] + "\n")
  else:
    textFileTest.write("data/obj/" + imgList[i] + "\n")

textFileTest.close()
textFileTrain.close()