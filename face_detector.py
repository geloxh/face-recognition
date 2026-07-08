import cv2
import numpy as np
import pickle
from deepface import DeepFace

with open("ref_name.pkl", "rb") as f:
    ref_dictt = pickle.load(f)
    
with open('ref_embed.pkl', "rb") as f:
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

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    if process_this_frame:
        faces_names = []
        face_boxes = []
        
        try:
            results = DeepFace.represent(small_frame, model_name="Facenet", enforce_detection=False)
            faces = DeepFace.extract_faces(small_frame, enforce_detection=False)
            
            for i, result in enumerate(results):
                face_encoding = np.array(result['embedding'])
                face_area = faces[i]["facial_area"]
                
                distances = [np.linalg.norm(face_encoding - e) for e in known_face_encodings]
                best_index = np.argmin(distances)
                name = known_face_names[best_index] if distances[best_index] < 10 else "Unknown"
                
                face_names.append(name)
                face_boxes.append(face_area)
                
        except:
            pass
        
process_this_frame = not process_this_frame
 
for face_area, name in zip(face_boxes, faces_names):
	left = face_area["x"] * 4
	top = face_area["y"] * 4
	right = (face_area["x"] + face_area["w"]) * 4
	bottom = (face_area["y"] + face_area["h"]) * 4
 
	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
 