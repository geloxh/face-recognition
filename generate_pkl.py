import pickle
import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis

FACES_FOLDER = "faces"

app = FaceAnalysis(name="buffalo_sc")
app.prepare(ctx_id=0, det_size=(640, 640))

def generate_encodings():
    known_encodings = []
    known_names = []

    if not os.path.exists(FACES_FOLDER):
        print(f"❌ '{FACES_FOLDER}' folder not found!")
        return

    persons = os.listdir(FACES_FOLDER)
    if not persons:
        print(f"❌ '{FACES_FOLDER}' folder is empty!")
        return

    print(f"📁 Found {len(persons)} person(s): {persons}\n")

    for person_name in persons:
        person_folder = os.path.join(FACES_FOLDER, person_name)
        if not os.path.isdir(person_folder):
            continue

        images = os.listdir(person_folder)
        print(f"👤 Processing: {person_name} ({len(images)} image(s))")

        for image_file in images:
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"   ⚠️  Skipping: {image_file}")
                continue

            image_path = os.path.join(person_folder, image_file)
            try:
                img = cv2.imread(image_path)
                faces = app.get(img)
                if faces:
                    known_encodings.append(faces[0].embedding)
                    known_names.append(person_name)
                    print(f"   ✅ Encoded: {image_file}")
                else:
                    print(f"   ⚠️  No face found in: {image_file}")
            except Exception as e:
                print(f"   ❌ Error: {image_file}: {e}")

    if known_encodings:
        ref_dictt = {name: name for name in set(known_names)}
        embed_dictt = {}
        for name, enc in zip(known_names, known_encodings):
            embed_dictt.setdefault(name, []).append(enc)

        with open("ref_name.pkl", "wb") as f:
            pickle.dump(ref_dictt, f)
        with open("ref_embed.pkl", "wb") as f:
            pickle.dump(embed_dictt, f)

        print(f"\n✅ Done! {len(known_encodings)} face(s) saved.")
    else:
        print("\n❌ No faces encoded.")

if __name__ == "__main__":
    generate_encodings()
