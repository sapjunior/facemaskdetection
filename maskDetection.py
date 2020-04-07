import cv2
import onnxruntime as rt
import numpy as np
from utils import *

class maskDetection():
    def __init__(self, modelFile, faceThreshold = 0.9):
        sessOptions = rt.SessionOptions()
        sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sessOptions.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        self.inferenceSession = rt.InferenceSession(modelFile, sessOptions)
        self.faceThreshold = faceThreshold
        self.anchorsCache = {}

        self.featStride = [32, 16, 8]
        self.retinaAnchors = {
            32: np.array([[-248.,-248.,263.,263.],[-120.,-120.,135.,135.]], dtype=np.float32), 
            16: np.array([[-56.,-56.,71.,71.],[-24.,-24.,39.,39.]], dtype=np.float32), 
            8: np.array([[-8.,-8.,23.,23.],[0.,0.,15.,15.]], dtype=np.float32)
        }

    def detect(self, inputImage):
        resizedImage, resizedRatio = resize2TargetSize(inputImage)
        inputImageTensor = resizedImage.transpose(2,0,1)[np.newaxis]
        inferenceOutputs = self.inferenceSession.run([], {"input": inputImageTensor})
        faceOutputs = self.infer2Box(inferenceOutputs, resizedImage.shape[0:2], resizedRatio)
        return faceOutputs

    def infer2Box(self,outputs, imageSize, resizedRatio):
        faceBoxes = []
        faceScores = []
        maskScores = []

        featIdx = 0

        for currentStride in self.featStride:
            scaleFaceScores = outputs[featIdx].transpose(0,2,3,1).reshape(-1, 1)
            scaleFaceBoxes = outputs[featIdx+1]
            scaleMaskScores = outputs[featIdx+2].transpose(0,2,3,1).reshape(-1, 1)
            featIdx+=3
            
            height, width = scaleFaceBoxes.shape[2], scaleFaceBoxes.shape[3]
            if (height, width, currentStride) in self.anchorsCache:
                anchors = self.anchorsCache[(height, width, currentStride)]
            else:
                anchors = generateAnchor(height, width, currentStride, self.retinaAnchors[currentStride])
                anchors = anchors.reshape((height*width*2, 4))
                self.anchorsCache[(height, width, currentStride)] = anchors

            scaleFaceBoxes = scaleFaceBoxes.transpose(0,2,3,1).reshape(-1, 4)

            scaleFaceBoxes = bboxTransform(anchors, scaleFaceBoxes)
            scaleFaceBoxes = boundBoxes(scaleFaceBoxes, imageSize[0], imageSize[1])

            faceBoxes.append(scaleFaceBoxes)
            faceScores.append(scaleFaceScores)
            maskScores.append(scaleMaskScores)
        
        faceBoxes = np.vstack(faceBoxes)
        faceScores = np.vstack(faceScores)
        maskScores = np.vstack(maskScores)

        keepIdx = np.where(faceScores.ravel() >= self.faceThreshold)[0]
        faceBoxes = faceBoxes[keepIdx,:]
        faceScores = faceScores[keepIdx]
        maskScores = maskScores[keepIdx]

        faceBoxes /= resizedRatio

        faces = np.hstack((faceBoxes[:,0:4], faceScores)).astype(np.float32)
        keepIdx = nms(faces,0.4)
        faces = np.hstack((faces, maskScores))
        faces = faces[keepIdx, :]

        return faces
