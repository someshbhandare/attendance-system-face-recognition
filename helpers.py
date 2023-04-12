import os
import cv2
import face_recognition
from datetime import datetime


# function for getting images_list & image_names
def get_names_images(path):
    allowed_formats = (".jpg", ".png")
    image_list = [img for img in os.listdir(path) if os.path.splitext(img)[1] in allowed_formats]

    images = []
    names = []
    prns = []
    for img in image_list:
        currImg = cv2.imread(f'{path}/{img}') #opening image
        currImg = cv2.cvtColor(currImg, cv2.COLOR_BGR2RGB)
        images.append(currImg)

        prn, name = os.path.splitext(img)[0].split('_')
        names.append(name)
        prns.append(prn)

    return prns, names, images


# function for finding encodings
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img, num_jitters=5, model="cnn")[0]
        encode_list.append(encode)
    return encode_list


# mark attendance
def mark_attendance(student):

    file = f'attendance_sheets/{datetime.now().strftime("%d-%m-%Y")}'
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.writelines('PRN, Name, Time, Date\n')

    with open(file, 'r+') as f:
        myDataList = f.readlines()
        myDataList = myDataList[1:]
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if student['prn'] not in nameList:
            time = datetime.now().strftime("%H:%M:%S %p")
            date = datetime.now().strftime("%d-%m-%Y")
            f.writelines(f"{student['prn']}, {student['name']}, {time}, {date}\n")
