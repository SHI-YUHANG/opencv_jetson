import cv2 
import matplotlib.pyplot as plt
import numpy as np
# img = cv2.imread('opencv/picture/WhiteDog.webp')
# cv2.imshow("image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('opencv/picture/WhiteDog.webp',cv2.IMREAD_GRAYSCALE)
cv_show("dog",img)

# print(img.shape)