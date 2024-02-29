import cv2

def detect_faces_in_image(image_path):
    # 加载预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 读取照片
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # 在检测到的人脸上绘制矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 显示结果
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)  # 等待直到有键被按下
    cv2.destroyAllWindows()

# 使用函数的示例：
image_path = '/Users/shiyuhang/Desktop/code/opencv/image/tes/lenal.webp' # 替换为你的图片地址
detect_faces_in_image(image_path)
