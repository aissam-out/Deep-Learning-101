import cv2
import numpy as np
import matplotlib.pyplot as plt

# the test image
image = cv2.imread('images/family.jpg')

# Extract the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('haar_detectors/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar_detectors/haarcascade_eye.xml')


def face_eye_detector(image, eye=True, display=True):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    denoised_image = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    image_b = np.copy(denoised_image)
    gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    for (x,y,w,h) in faces:
        cv2.rectangle(image_b,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image_b[y:y+h, x:x+w]
        if eye == True :
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    if display == True :
        # Display the image with the detections
        plt.figure()
        plt.title('Image with Face and eye Detections')
        plt.imshow(image_b)
        plt.show()

    return image_b, roi_color, faces

face_eye_detector(image)
