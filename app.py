import cv2
import keyboard


VIDEOCAPTURE = cv2.VideoCapture(0)
VIDEOCAPTURE.set(3, 640)  # set Width
VIDEOCAPTURE.set(4, 480)  # set Height
FACEMODEL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_detection():
    while not keyboard.is_pressed("q":
        placeholder, frame = VIDEOCAPTURE.read()
        frame = cv2.flip(frame, 1)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACEMODEL.detectMultiScale(
            grayscale, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(5, 5)
        )

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            grayscale[y : y + h, x : x + w]
            frame[y : y + h, x : x + w]

        cv2.imshow("Face Detection", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    face_detection()
