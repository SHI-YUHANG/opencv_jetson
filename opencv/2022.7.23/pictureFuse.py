import cv2 
import matplotlib.pyplot as plt
import numpy as np

img_Wdog = cv2.imread("opencv/picture/WhiteDog.webp")
img_Ydog = cv2.imread("opencv/picture/YellowDog.webp")
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_Ydog2 = cv2.resize(img_Ydog,(375,500))
res = cv2.addWeighted(img_Ydog2,0.4,img_Wdog,0.6,0)
cv_show("xxxx",res)
# cv_show("WhiteDog",img_Wdog2)