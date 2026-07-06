import face_recognition
import pickle
import os

# ── Config ──
FACES_FOLDER = "faces" # folder containing subfolders per person
OUTPUT_FILE = "ref_name.pkl"  # output pickle file
# ──

def generate_encodings():
    known_encodings = []
    known_names = []
    errors = []

    # check if faces folder exists
    if not os.path.exists(FACES_FOLDER):
        print(f"❌ '{FACES_FOLDER}' folder not found!")
        print(f"   Create it and add subfolders named after each person.")
        sys.exit(1)

    persons = os.listdir(FACES_FOLDER)

    if not persons:
        print(f"❌ '{FACES_FOLDER}' folder is empty!")
        print(f"   Add subfolders with face images inside.")
        sys.exit(1)

    print(f"📁 Found {len(persons)} person(s): {persons}\n")

    for person_name in persons:
        person_folder = os.path.join(FACES_FOLDER, person_name)

        # skip if not a folder
        if not os.path.isdir(person_folder):
            continue

        images = os.listdir(person_folder)
        print(f"👤 Processing: {person_name} ({len(images)} image(s))")

        for image_file in images:
            image_path = os.path.join(person_folder, image_file)

            # only process image files
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"   ⚠️  Skipping non-image file: {image_file}")
                continue

            try:
                # load and encode the face
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    print(f"   ✅ Encoded: {image_file}")
                else:
                    print(f"   ⚠️  No face found in: {image_file}")
                    errors.append(image_path)

            except Exception as e:
                print(f"   ❌ Error processing {image_file}: {e}")
                errors.append(image_path)

    # save to pkl
    if known_encodings:
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump((known_encodings, known_names), f)

        print(f"\n✅ Done! {len(known_encodings)} face(s) saved to '{OUTPUT_FILE}'")
    else:
        print("\n❌ No faces encoded. Check your images and try again.")

    # show errors summary
    if errors:
        print(f"\n⚠️  {len(errors)} image(s) had issues:")
        for err in errors:
            print(f"   - {err}")

if __name__ == "__main__":
    generate_encodings()