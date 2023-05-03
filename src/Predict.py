import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import pickle

# Load the model
model = tf.keras.models.load_model('asl-ml/src/final_asl3.h5')

# Load the StandardScaler
with open('asl-ml/src/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the LabelEncoder
with open('asl-ml/src/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Initialize the Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break

    # Detect hand
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        # Preprocess the input
        imgResized = np.array(lmList).flatten()
        imgResized = scaler.transform([imgResized])

        # Predict the gesture
        predictions = model.predict(imgResized)
        class_index = np.argmax(predictions[0])
        class_label = encoder.inverse_transform([class_index])[0]

        # Display the prediction on the image
        cv2.putText(img, class_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the image
    cv2.imshow("Image", img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()