import cv2
import pickle
from deepface import DeepFace

name = input("Enter the Name: ")
ref_id = input("Enter the ID: ")

try:
    with open("ref_name.pkl", "rb") as f:
        ref_dictt = pickle.load(f)
except:
    ref_dictt = {}

ref_dictt[ref_id] = name
with open("ref_name.pkl", "wb") as f:
    pickle.dump(ref_dictt, f)

try:
    with open("ref_embed.pkl", "rb") as f:
        embed_dictt = pickle.load(f)
except:
    embed_dictt = {}

print("Press 's' to capture face (5 times), 'q' to quit.")

captures = 0
webcam = cv2.VideoCapture(0)

while captures < 5:
    check, frame = webcam.read()
    if not check:
        break

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
        face_encoding = result[0]["embedding"]

        if ref_id in embed_dictt:
            embed_dictt[ref_id].append(face_encoding)
        else:
            embed_dictt[ref_id] = [face_encoding]

        captures += 1
        print(f"Captured {captures}/5")

    elif key == ord('q'):
        print("Cancelled.")
        break

webcam.release()
cv2.destroyAllWindows()

with open("ref_embed.pkl", "wb") as f:
    pickle.dump(embed_dictt, f)

print("Done! Face data saved.")