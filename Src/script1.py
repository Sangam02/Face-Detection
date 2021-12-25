import cv2,glob

all_img=glob.glob("..\\Face Detection\\Assets\\*.jpg")

detect = cv2.CascadeClassifier("..\\Face Detection\\Lib\\haarcascade_frontalface_default.xml")


for image in all_img:
    img =cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(gray,1.3,5)


    for (x, y, w, h) in faces:
        final_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Face Detection", final_img)

    cv2.waitKey(5000)
    cv2.destroyAllWindows()