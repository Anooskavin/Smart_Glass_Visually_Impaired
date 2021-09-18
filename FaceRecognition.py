import face_recognition
import cv2
import numpy as np
from espeak import espeak

video_capture = cv2.VideoCapture(0)
a_image = face_recognition.load_image_file("ash.jpg")
a_face_encoding = face_recognition.face_encodings(a_image)[0]
b_image = face_recognition.load_image_file("b.jpg")
b_face_encoding = face_recognition.face_encodings(b_image)[0]

known_face_encodings = [
    a_face_encoding,
    b_face_encoding
]
known_face_names = [
    "A ",
    "b"
]
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                #espeak.synth("Hey I Can Identify your face")
                #espeak.synth("Your name is")
                #espeak.synth(name)
                
                

            

