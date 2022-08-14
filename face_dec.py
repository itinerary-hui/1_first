import cv2

def dec_face():
    face_cascade = cv2.CascadeClassifier('D:\BaiduNetdiskDownload\haarshare/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('D:\BaiduNetdiskDownload\haarshare/haarcascade_eye_tree_eyeglasses.xml')
    cap = cv2.VideoCapture(0) #调用摄像头
    while(True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1) #将人脸用绿色方框起来
            cv2.putText(img, 'Trump', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 2) #显示人脸信息
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1) #将眼睛用绿色方框圈起来
        cv2.imshow('face dectect',img) #窗口的名字为人脸检测
        if cv2.waitKey(30) & 0xFF == ord('q'): #按键'q'关闭窗口
            break
    cap.release() #释放摄像头
    #out.release()
    cv2.destroyAllWindows()

dec_face()