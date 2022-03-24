import numpy as np
import cv2
import tensorflow as tf
import time
from utils.model_architecture import get_model


# Model list
model_list = ["Model/CNN_model_V2.h5"]    

print("[INFO] Loading CNN model...")
model = get_model()
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

model.load_weights(model_list[-1])

print("[INFO] Loading HaarCascade model...")
face_detection = cv2.CascadeClassifier("utils/Haar_Cascade/haar_cascade_face_detection.xml")

print("[INFO] Loading Camera/Video...")
camera = cv2.VideoCapture(0)

settings = {
    'scaleFactor': 1.3,
    'minNeighbors': 5,
    'minSize': (50, 50)
}

while True:
    ret, img = camera.read()
    
    origin_h, origin_w = img.shape[:2]
    
    start = time.time()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detected = face_detection.detectMultiScale(gray, **settings)

    try:
        for (x, y, w, h) in detected:
        
            cv2.rectangle(img, (x, y), (x + w, y + h), (245, 135, 66), 2)
            cv2.rectangle(img, (x, y-18), (x + w, y), (245, 135, 66), -1)

            face = gray[int(y):(int(y) + int(h)), int(x):(int(x) + int(w))]
            face = cv2.resize(face, (48, 48))

            img_pixels = np.array(face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)

            img_pixels = img_pixels / 255.0

            predictions = model.predict(img_pixels)

            pred = predictions.argmax()

            score = int(np.amax(predictions) * 100)

            state = labels[pred]

            cv2.putText(img, state + ": %.0f" % score + "%",
                        (x + 2, y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    except Exception as ex:
        print(ex)
        
    # Calc FPS
    end = time.time()
    totalTime = end - start
    if totalTime == 0:
        totalTime = 1
    fps = 1 // totalTime
    
    cv2.putText(img, "FPS: " + str(fps), (15, int(origin_h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Facial Expression', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # writer.writeFrame(img)
    if cv2.waitKey(5) != -1:
        break

camera.release()
cv2.destroyAllWindows()