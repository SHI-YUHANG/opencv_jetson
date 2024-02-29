import cv2
from cv2 import threshold 
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
img = cv2.imread("opencv/picture/zaodian.webp")

#均值滤波
blur = cv2.blur(img,(3,3))
# cv_show('blur',blur)

#高斯滤波
aussian = cv2.GaussianBlur(img,(5,5),1)
# cv_show('aussian',aussian)

#中值滤波
median = cv2.medianBlur(img,5)
# cv_show('median',median)

res = np.hstack((img,blur,aussian,median))
print(res)
cv_show("xxx",res)