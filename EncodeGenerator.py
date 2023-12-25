import cv2
import face_recognition
import pickle
import os
# importing images
folderImg = "Images"
pathList = os.listdir(folderImg)
# print(pathList)
imgList = []
studentId = []
# print(pathList)
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderImg, path)))

    # removing .png/.jpg from the name of the images
    # print(os.path.splitext(path)[0])
    studentId.append(os.path.splitext(path)[0])
print(studentId)


def findEncodings(imgList):
    encodeList = []
    for img in imgList:
        # img = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1, interpolation= cv2.INTER_AREA)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding started....")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentId]
print(encodeListKnown)
print("Encoding completed.")


# saving the ecodings in a file
file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File saved")