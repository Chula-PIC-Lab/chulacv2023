## wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
## bash Miniforge3-Linux-aarch64.sh
## mamba create -n cv2023 python=3.9
## mamba activate cv2023
## mamba install onnx protobuf numpy pip six fastapi uvicorn python-multipart
## pip install opencv-python # need to install from pip due to QT dependencies on arm64

## ONNXRuntime https://elinux.org/Jetson_Zoo#ONNX_Runtime
## wget https://nvidia.box.com/shared/static/jmomlpcctmjojz14zbwa12lxmeh2h6o5.whl -O onnxruntime_gpu-1.11.0-cp39-cp39-linux_aarch64.whl
## pip install onnxruntime_gpu-1.11.0-cp39-cp39-linux_aarch64.whl

## Download model
## wget https://piclab.ai/classes/cv2023/raccoons.onnx

import cv2
import onnxruntime as rt
import numpy as np

####
sessOptions = rt.SessionOptions()
sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL 
raccoonModel = rt.InferenceSession('raccoons.onnx', sessOptions)
####
inputStream = cv2.VideoCapture(0)

while True:
    isImageValid, inputImage = inputStream.read()
    
    if isImageValid:
        ### Pre-processing ###
        inputTensor = cv2.resize(inputImage, (320,320))
        inputTensor = (inputTensor - [103.53, 116.28, 123.675]) / [57.375, 57.12, 58.395]
        inputTensor = inputTensor.transpose(2,0,1)[np.newaxis].astype(np.float32) #NCHW
        ### Inference ###
        outputBoxes, outputLabels = raccoonModel.run([], {'input': inputTensor})

        ### Post-processing (rescale) ###
        outputImage = inputImage.copy()
        ratioH, ratioW = inputImage.shape[0] / 320, inputImage.shape[1] / 320
        rescaleOutputBoxes = outputBoxes * [ratioW, ratioH, ratioW, ratioH, 1]

        ### Draw and display output ###
        for boxData in zip(rescaleOutputBoxes[0], outputLabels[0]):
            prob  = boxData[0][4]
            print(boxData)
            if prob > 0.75:
                x1,y1,x2,y2 = boxData[0][0:4].astype(np.int32)
                label = boxData[1]
                cv2.rectangle(outputImage, (x1,y1), (x2,y2), (0,255,0), 3)
        
        cv2.imshow("Output", outputImage)
        cv2.waitKey(1)
    else:
        print('Cannot open camera')
        break

inputStream.release()
