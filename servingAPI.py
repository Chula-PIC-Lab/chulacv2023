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
## wget https://piclab.ai/classes/cv2023/chestxray.onnx

## Start API
## uvicorn servingAPI:app --host 0.0.0.0 --port 8500

import cv2
import onnxruntime as rt
import numpy as np
from fastapi import FastAPI, File
####
sessOptions = rt.SessionOptions()
sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL 
chestxrayModel = rt.InferenceSession('chestxray.onnx', sessOptions, providers=[ 'CUDAExecutionProvider', 'CPUExecutionProvider'])
####
app = FastAPI()

def decodeByte2Numpy(inputImage):
    outputImage = np.frombuffer(inputImage, np.uint8)
    outputImage = cv2.imdecode(outputImage, cv2.IMREAD_COLOR)
    return outputImage

@app.post('/chestxray')
def readmeBackend(image: bytes = File(...)):
    try:
        inputImage = decodeByte2Numpy(image)
        inputImage = cv2.resize(cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB), (224,224))
        inputTensor = ((inputImage / 255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        inputTensor = inputTensor.transpose(2,0,1)[np.newaxis].astype(np.float32)

        outputTensor = chestxrayModel.run([], {'input': inputTensor})[0]

        outputDict = {'label': int(np.argmax(outputTensor)) }
        return outputDict    
    except Exception as e:
        print(e)
        return {'status':'INVALID_IMAGE_FILE'}