import cv2
from cv2 import threshold 
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("opencv/picture/YellowAndWhiteDog.webp",cv2.IMREAD_GRAYSCALE)
#大于127的部分设为255 否则取0
ret,threshold1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#小于127的部分设为255 否则取0
ret,threshold2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#大于127的部分设为255 否则不变
ret,threshold3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
#大于127的部分保持不变 否则设为0
ret,threshold4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
#上面的反转
ret,threshold5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv_show("xxx",img)