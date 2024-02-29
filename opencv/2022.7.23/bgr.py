import cv2 
import matplotlib.pyplot as plt
import numpy as np
def test(name,img):
    dog = img[0:500,0:500]
    # b,g,r = cv2.split(dog)
    # x = cv2.merge((b,g,r))
    img_copy = dog.copy()
    img_copy[:,:,0] = 0
    img_copy[:,:,1] = 0
    cv2.imshow("R",img_copy)
    cv2.waitKey(0)

    img_copy = dog.copy()
    img_copy[:,:,0] = 0
    img_copy[:,:,2] = 0
    cv2.imshow("G",img_copy)
    cv2.waitKey(0)

    img_copy = dog.copy()
    img_copy[:,:,1] = 0
    img_copy[:,:,2] = 0
    cv2.imshow("B",img_copy)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
img = cv2.imread("opencv/picture/YellowDog.webp")
test("dog",img)
