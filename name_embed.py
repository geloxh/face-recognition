import sys

sys.path.append()
import cv2
import face_recognition
import pickle

name=input("Enter the Name: ")
ref_id=input("Enter the ID: ")

try:
    f=open("ref_name.pk1", "rb")
    
    ref_dictt=pickle.load(f)
    f.close()
except:
    ref_dictt={}
    ref_dictt[ref_id]=name
    
f=open("ref_name.pk1", "wb")
pickle.dump(ref_dictt, f)
f.close()

try:
    f=open("ref_embed.pk1", "rb")
    embed_dictt=pickle.load(f)
    f.close()

except:
    embed_dictt={}
    
for i in range(5):
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        
        check, frame = webcam.read()
    