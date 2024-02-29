import cv2
from cv2 import threshold 
import matplotlib.pyplot as plt
import numpy as np
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
img = cv2.imread("opencv/picture/shiyuhang.webp")


#腐蚀操作
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 2) 

#膨胀操作
kernel = np.ones((3,3),np.uint8)
dilate = cv2.dilate(erosion,kernel,iterations = 2)

res = np.hstack((img,dilate))
cv_show("xxx",res)  