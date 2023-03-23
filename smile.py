import cv2
import tkinter as tk
from tkinter import messagebox
import webbrowser

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        if len(smiles) == 0:
            messagebox.showinfo("Hey!", "If you smile I will give you a gift. Please smile!")
        else:
            messagebox.showinfo("Thank you for smiling!", "Have a nice day!")
            webbrowser.open('https://t3.ftcdn.net/jpg/03/17/02/66/360_F_317026621_gxBKhW9g1aUgU0kMO5q2ROmfzDmN6zvd.jpg')
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
