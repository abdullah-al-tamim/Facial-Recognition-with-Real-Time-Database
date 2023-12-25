import os
import cv2
import pickle
import numpy
import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("Resources/background.png")

# importing modes
folderModePath = "Resources/Modes"
modePathList = os.listdir(folderModePath)
imgModeList = []

# print(modePathList)
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

#load the encoding file
print("Loading encoded file...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("File loaded")
print(studentIds)


while True:
    success, img = cap.read()
    # making the img small and converting BGR 2 RGB
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    #encoding the face from the frame
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)


    imgBackground[162: 162+480, 55:55+640] = img
    imgBackground[44: 44+633, 808:808+414] = imgModeList[1]

    #
    for encodeFace, faceloc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("Matches: ", matches)
        print("Face Distance: ", faceDiffs)
    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)