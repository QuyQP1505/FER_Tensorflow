import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from utils.model_architecture import get_model
import imutils
import cv2


print("[INFO] Loading CNN model...")
model = get_model()
labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.load_weights(model_list[-1])

print("[INFO] Loading ResNetSSD model ...")
prototxt_file = "utils/ResNet_SSD/deploy.prototxt"
caffemodel_file = "utils/ResNet_SSD/Res10_300x300_SSD_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt_file, caffemodel_file)

print("[INFO] Loading camera/video ...")
camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()
    
    if isinstance(image,type(None)): break
    
    start = time.time()
    
    base_img = image.copy()
    original_size = base_img.shape
    target_size = (700, 400)
    image = cv2.resize(image, target_size)
    aspect_ratio_x = (original_size[1] / target_size[1])
    aspect_ratio_y = (original_size[0] / target_size[0])
            
    origin_h, origin_w = image.shape[:2]

    img_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    detector.setInput(img_blob)
    
    detections = detector.forward()

    try:
        for i in range(0, detections.shape[2]):
        
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:

                bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])

                x_start, y_start, x_end, y_end = bounding_box.astype('int')

                face = image[int(y_start):int(y_end), int(x_start):int(x_end)]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) #transform to gray scale
                face = cv2.resize(face, (48, 48))

                img_pixels = np.array(face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)

                img_pixels = img_pixels / 255.0

                predictions = model.predict(img_pixels)

                pred = predictions.argmax()

                score = int(np.amax(predictions) * 100)

                emotion = labels[pred]

                # bounding box
                cv2.rectangle(image, (x_start, y_start), (x_end, y_end),(0, 0, 255), 2)
                cv2.rectangle(image, (x_start, y_start-18), (x_end, y_start), (0, 0, 255), -1)
                cv2.putText(image, emotion + ": %.0f" % score + "%"
                            , (x_start+2, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    except Exception as ex:
        print(ex)

    # Calc FPS
    end = time.time()
    totalTime = end - start
    if totalTime == 0:
        totalTime = 1
    
    fps = 1 // totalTime
    cv2.putText(image, "FPS: " + str(fps), (15, int(origin_h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Facial Expression', image)
    
    if cv2.waitKey(5) != -1:
        break

camera.release()
cv2.destroyAllWindows()