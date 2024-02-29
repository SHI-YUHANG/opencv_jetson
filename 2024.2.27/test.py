import cv2

# 加载预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开视频流
cap = cv2.VideoCapture(0)  # 0 通常是默认的摄像头
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # 读取视频流中的帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # 在检测到的人脸上绘制矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 显示结果
    cv2.imshow('Face Detection', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
