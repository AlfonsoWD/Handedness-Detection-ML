import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load the trained model
model = joblib.load("hand_classifier.pkl")  # Ensure this file is in the same directory

# Initialize MediaPipe for up to 2 hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, 
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(1) # Use 1 for front camera, 0 for back camera (0 if only one camera)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press 'q' or close the window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 63 features from hand landmarks
            feature_row = []
            for lm in hand_landmarks.landmark:
                feature_row.extend([lm.x, lm.y, lm.z])

            # Get prediction + probability
            proba = model.predict_proba([feature_row])[0]
            label_index = np.argmax(proba)
            confidence = proba[label_index]
            label = "Left" if label_index == 0 else "Right"

            # Display result on frame
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[0].x * w)
            y = int(hand_landmarks.landmark[0].y * h)
            cv2.putText(frame, f"{label} ({confidence * 100:.1f}%)", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    else:
        cv2.putText(frame, "No hand detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Custom Handedness Model", frame)

    if cv2.getWindowProperty("Custom Handedness Model", cv2.WND_PROP_VISIBLE) < 1 or \
       cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


