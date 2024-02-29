import os
import cv2
import numpy as np
from PIL import Image

# 初始化人脸检测器和人脸识别器
# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# 准备训练数据
# 假设你的训练数据存放在 'dataset' 目录下，每个子目录代表一个人
# 每个子目录的名字为标签，里面是该人的照片
def get_images_and_labels(path):
    # 定义支持的图像文件扩展名列表
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1].lower() in valid_extensions]
    face_samples = []
    ids = []

    for image_path in image_paths:
        PIL_img = Image.open(image_path).convert('L')  # 转换为灰度图
        img_numpy = np.array(PIL_img, 'uint8')

        # 假设文件名格式为 "personID_something.jpg"
        # 你需要根据你的文件命名规则来调整下面的代码
        id = int(os.path.split(image_path)[-1].split("_")[0])
        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return face_samples, ids


faces, ids = get_images_and_labels('/Users/shiyuhang/Desktop/code/opencv/image/syh')
face_recognizer.train(faces, np.array(ids))

# 保存训练模型
face_recognizer.write('trainer.yml')
