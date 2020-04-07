import cv2
from maskDetection import *

FACE_THESH = 0.75
MASK_THESH = 0.25
CAM_NO = 0

detector = maskDetection('maskDet_opt.onnx', faceThreshold = FACE_THESH)

inputStream = cv2.VideoCapture(CAM_NO)
inputStream.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
inputStream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
inputStream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    isFrameOK, inputFrame = inputStream.read()
    if isFrameOK:
        outputFaces = detector.detect(inputFrame)
        for faceIdx in range(outputFaces.shape[0]):
            faceBox = outputFaces[faceIdx, 0:4].astype(np.int)
            faceProb = outputFaces[faceIdx, 4]
            maskProb = outputFaces[faceIdx, 5]

            if maskProb <= MASK_THESH:
                boxColor = (0,0,255)
            else:
                boxColor = (0,255,0)

            cv2.rectangle(inputFrame, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), boxColor, 2)

        cv2.imshow("out", inputFrame)
        if ord('q') == cv2.waitKey(1):
            break
    else:
        print('Cannot connect to camera')
        break
inputStream.release()