import cv2
import face_recognition
import numpy as np
from helpers import get_names_images, find_encodings, mark_attendance

if __name__ == "__main__":
    # path for images folder
    path = "images"

    # getting images and their names
    prns, names, images = get_names_images(path)
    print(names)


    # encodings of known faces
    knownFacesEncodings = find_encodings(images)
    print("Encoding completed")
    print(len(knownFacesEncodings))

    # capture images throught webcam
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        frameS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

        # face locations & encodings of faces in current frame
        facesCurrFrame = face_recognition.face_locations(frameS)
        encodingsCurrFrame = face_recognition.face_encodings(frameS, facesCurrFrame)
        print(len(facesCurrFrame))

        for faceLoc, faceEncode in zip(facesCurrFrame, encodingsCurrFrame):
            matches = face_recognition.compare_faces(knownFacesEncodings, faceEncode, tolerance=0.5)
            faceDis = face_recognition.face_distance(knownFacesEncodings, faceEncode)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            # default name
            name = "unknown"
            print(matches)

            # if matches[matchIndex]:
            student = {}
            if faceDis[matchIndex] < 0.45:
                prn = prns[matchIndex]
                nm = names[matchIndex].split(" ")[0].upper()
                name = f'{prn}_{nm}'
                student['prn'] = prn
                student['name'] = names[matchIndex]


            # +++++++ display result
            print(name)
            top, right, bottom, left = faceLoc
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            width, height = abs(left - right), abs(top-bottom)
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255, 0), 2)
            cv2.rectangle(frame, (left, bottom-30), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            if name != 'unknown':
                mark_attendance(student)

        # captu
        cv2.imshow("webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break



