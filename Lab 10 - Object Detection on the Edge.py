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

def resize(image, input_size):
  shape = image.shape

  ratio = float(shape[0]) / shape[1]
  if ratio > 1:
      h = input_size
      w = int(h / ratio)
  else:
      w = input_size
      h = int(w * ratio)
  scale = float(h) / shape[0]
  resized_image = cv2.resize(image, (w, h))
  det_image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
  det_image[:h, :w, :] = resized_image
  return det_image, scale

####
sessOptions = rt.SessionOptions()
sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL 
raccoonModel = rt.InferenceSession('raccoons.onnx', sessOptions)
####
inputStream = cv2.VideoCapture(1)

while True:
    isImageValid, inputImage = inputStream.read()
    
    if isImageValid:
        ### Pre-processing ###
        image, scale = resize(inputImage, 640)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))[::-1]
        image = image.astype('float32') / 255
        image = image[np.newaxis, ...]
        ### Inference ###
        outputs = raccoonModel.run([], {'images': image})

        ### Post-processing (rescale) ###
        outputs = np.transpose(np.squeeze(outputs[0]))

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_indices = []

        # Iterate over each row in the outputs array
        for i in range(outputs.shape[0]):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= 0.25:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                image, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((image - w / 2) / scale)
                top = int((y - h / 2) / scale)
                width = int(w / scale)
                height = int(h / scale)

                # Add the class ID, score, and box coordinates to the respective lists
                class_indices.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.7)

        # Iterate over the selected indices after non-maximum suppression
        nms_outputs = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_indices[i]
            nms_outputs.append([*box, score, class_id])

        ### Draw and display output ###
        for output in nms_outputs:
            x, y, w, h, score, index = output
            cv2.rectangle(inputImage, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        
        cv2.imshow("Output", inputImage)
        cv2.waitKey(1)
    else:
        print('Cannot open camera')
        break

inputStream.release()
