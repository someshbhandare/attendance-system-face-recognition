import face_recognition
import os
import cv2
import pymongo

# path
path = 'images'
allowed_formats = ('.jpg', '.png')
files_list = [i for i in os.listdir(path) if os.path.splitext(i)[1] in allowed_formats]

# getting all images & their names
images = []
names = []
for imgName in files_list:
    curImg = cv2.imread(f'{path}/{imgName}')
    images.append(curImg)
    names.append(os.path.splitext(imgName)[0])
print(names)

# function for finding encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img, num_jitters=10)[0]
        encodeList.append(encode)
    return encodeList

# knownFaces = findEncodings(images)
print("Encoding done")


client = pymongo.MongoClient("mongodb+srv://someshbhandare:somesh38@cluster0.47kwmrw.mongodb.net/?retryWrites=true&w=majority")
db = client.get_database("TY-BTech")
collection = db.get_collection("FaceEncodings")
# collection.insert_one({
#     "name": "rohit",
#     "encoding": knownFaces[1].tolist()
# })

data = collection.find()
print(data.next().get("name"))
