import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_drawing = mp.solutions.drawing_utils

# CSV setup
csv_file = 'hand_data.csv'
write_header = not os.path.exists(csv_file)

# Open CSV file
csvfile = open(csv_file, mode='a', newline='')
csvwriter = csv.writer(csvfile)
if write_header:
    header = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
    header.append('label')
    csvwriter.writerow(header)

# Webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press 'l' to save LEFT hand, 'r' to save RIGHT hand, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_row = []
            for lm in hand_landmarks.landmark:
                landmark_row.extend([lm.x, lm.y, lm.z])

            # Wait for user input to label
            key = cv2.waitKey(1) & 0xFF
            if key == ord('l'):
                csvwriter.writerow(landmark_row + ['Left'])
                print("Saved: Left hand")
            elif key == ord('r'):
                csvwriter.writerow(landmark_row + ['Right'])
                print("Saved: Right hand")

    else:
        cv2.putText(frame, "No hand detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Data Collection - Press 'l' or 'r'", frame)

    if cv2.getWindowProperty("Data Collection - Press 'l' or 'r'", cv2.WND_PROP_VISIBLE) < 1 or \
       cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
csvfile.close()
cv2.destroyAllWindows()
print("Data collection ended.")
