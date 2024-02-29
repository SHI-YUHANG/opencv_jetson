import cv2

def detect_and_recognize_faces_in_video():
    # 加载预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 加载训练好的人脸识别模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('/Users/shiyuhang/Desktop/code/trainer.yml')  # 确保这里的路径是正确的
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        # 读取视频流中的帧
        ret, img = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 进行人脸检测
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        # 在检测到的人脸上绘制矩形框，并识别人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 识别人脸
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            # 可以根据id和confidence显示识别结果，例如：
            if confidence < 100:  # confidence越小表示匹配度越高
                text = f"ID: {id}, C: {confidence:.2f}"
            else:
                text = "Unknown"
            
            cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 显示结果
        cv2.imshow('Face Recognition', img)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

# 调用函数
detect_and_recognize_faces_in_video()
