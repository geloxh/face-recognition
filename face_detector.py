import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_sc")
app.prepare(ctx_id=0, det_size=(640, 640))

with open("ref_name.pkl", "rb") as f:
    ref_dictt = pickle.load(f)

with open("ref_embed.pkl", "rb") as f:
    embed_dictt = pickle.load(f)

known_face_encodings = []
known_face_names = []

for ref_id, embed_list in embed_dictt.items():
    for embed in embed_list:
        known_face_encodings.append(np.array(embed))
        known_face_names.append(ref_id)

video_capture = cv2.VideoCapture(0)

face_names = []
face_boxes = []
process_this_frame = True

THRESHOLD = 25.0  # tunable — lower = stricter

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if process_this_frame:
        face_names = []
        face_boxes = []

        faces = app.get(frame)
        for face in faces:
            face_encoding = np.array(face.embedding)
            distances = [np.linalg.norm(face_encoding - e) for e in known_face_encodings]

            if distances:
                best_index = np.argmin(distances)
                best_distance = distances[best_index]
                print(f"Best distance: {best_distance:.4f}")  # debug — check terminal
                name = known_face_names[best_index] if best_distance < THRESHOLD else "Unknown"
            else:
                name = "Unknown"

            face_names.append(name)
            face_boxes.append(face.bbox.astype(int))

    process_this_frame = not process_this_frame

    for bbox, name in zip(face_boxes, face_names):
        left, top, right, bottom = bbox
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        display_name = ref_dictt.get(name, name)
        cv2.putText(frame, display_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
