import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import pytesseract

detector = '/home/arun/Downloads/frozen_east_text_detection.pb'
width , height = 320 , 320
image = "/home/arun/Downloads/mug.jpg"
min_conf = 0.9

img = cv2.imread(image)
# cv2.imshow("mug",img)
# cv2.waitKey(0)

original = img.copy()
# print(img.shape)

H = img.shape[0]
W = img.shape[1]
# print(H, W)

proportion_W = W / float(width)
proportion_H = H / float(height)
# print(proportion_W,proportion_H)

img1 = cv2.resize(img, (width,height))
# print(img1.shape)

layer_name = ['feature_fusion/Conv_7/Sigmoid','feature_fusion/concat_3']

neural_network = cv2.dnn.readNet(detector)

blob = cv2.dnn.blobFromImage(img1,1.0, (width,height), swapRB = True, crop = False)

# print(blob.shape)

neural_network.setInput(blob)
scores, geometry = neural_network.forward(layer_name)

# print(scores)
# print(geometry)

# print(scores.shape)

rows, columns = scores.shape[2:]

boxes = []
confidences = []

def geometric_data(geometry, y):
  xData0 = geometry[0 , 0 , y]
  xData1 = geometry[0 , 1 , y]
  xData2 = geometry[0 , 2 , y]
  xData3 = geometry[0 , 3 , y]
  angles_data = geometry[0, 4, y]
  return angles_data, xData0, xData1, xData2, xData3

def geometric_calculation(angles_data, xData0, xData1, xData2, xData3):
  (offsetX, offsetY) = (x*4.0, y*4.0)
  angle = angles_data[x]
  cos = np.cos(angle)
  sin = np.sin(angle)
  h = xData0[x] + xData2[x]
  w = xData1[x] + xData3[x]

  endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
  endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))

  beginX = int(endX - w)
  beginY = int(endY - h)

  return beginX, beginY, endX, endY


for y in range(0, rows):
    data_scores = scores[0, 0, y]
    angles_data, xData0, xData1, xData2, xData3, = geometric_data(geometry, y)
    # print(data_scores)
    # print("-------")
    # print(angles_data,xData0,xData1,xData2,xData3)
    for x in range(0, columns):
        if data_scores[x] < min_conf:
            continue

        beginX, beginY, endX, endY = geometric_calculation(angles_data, xData0, xData1, xData2, xData3)

        confidences.append(data_scores[x])
        boxes.append((beginX, beginY, endX, endY))

# print(confidences)
# print(boxes)

detections = non_max_suppression(np.array(boxes), probs = confidences)
# print(detections)


#  Run the below command in the project directory in terminal
# !mkdir tessdata
# cd tessdata
# !wget https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata?raw=true

config_tesseract = "--tessdata-dir tessdata --psm 7"

margin = 3
img_copy = original.copy()
for (beginX, beginY, endX, endY) in detections:
    beginX = int(beginX * proportion_W)
    beginY = int(beginY * proportion_H)
    endX = int(endX * proportion_W)
    endY = int(endY * proportion_H)

    roi = img_copy[beginY - margin:endY + margin, beginX - margin:endX + margin]
    cv2.imshow(roi)
    text = pytesseract.image_to_string(roi, lang='eng', config=config_tesseract)
    print(text)

    cv2.rectangle(original, (beginX, beginY), (endX, endY), (0, 255, 100), 2)
cv2.imshow(original)
cv2.waitKey(0)

