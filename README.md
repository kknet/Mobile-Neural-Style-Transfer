# Mobile-Neural-Style-Transfer

PyTorch enables fast and flexible experimentation in deep learning research.
However, it is hard to deploy PyTorch model to the productive environment (e.g. smart phone).
The PyTorch model is converted to ONNX format which can be loaded with different frameworks (e.g. CoreML, Caffe).

## Environment

Ubnutu 16.04

Python 3.5

Mac 10.13.6

XCode 10.1

## Installation

    pip3 install torch torchvision onnx onnx_coreml

## Process

Download the pretrained models.
    
    python3 download_save_model.py

Convert the pretrained model to ONNX model.

    python3 pytorch_to_onnx.py saved_models/candy.pth saved_models/candy.onnx

Convert the ONNX model to CoreML model.

    python3 onnx_to_coreml.py saved_models/candy.onnx saved_models/candy.mlmodel

!! Remember to set input_names, output_names, image_input_names and image_output_names during the transformation process. It will generate the method **prediction(inputImage: CVPixelBuffer)-> modelOutPut** in XCode project.


## Referance
1. [PyTorch models](https://github.com/pytorch/examples)

2. [Building a Neural Style Transfer app on iOS with PyTorch and CoreML](https://medium.com/@alexiscreuzot/building-a-neural-style-transfer-app-on-ios-with-pytorch-and-coreml-76e00cd14b28)