# Mobile-Neural-Style-Transfer

PyTorch enables fast and flexible experimentation in deep learning research.
However, it is hard to deploy PyTorch model to the productive environment (e.g. smart phone).
The PyTorch model is converted to ONNX format which can be loaded with different frameworks (e.g. CoreML, Caffe).

## Environment

Ubnutu 16.04

Python 3.5

Mac 10.13.6

Xcode 10.1

## Installation

    pip3 install torch torchvision onnx onnx_coreml coremltools

## Model Conversion

### PyTorch -> ONNX -> CoreML

Download the pretrained models.
    
    python3 download_save_model.py

Convert the pretrained model to ONNX model.

    python3 pytorch_to_onnx.py saved_models/candy.pth saved_models/candy.onnx

Convert the ONNX model to CoreML model.

    python3 onnx_to_coreml.py saved_models/candy.onnx saved_models/candy.mlmodel

!! Remember to set input_names, output_names, image_input_names and image_output_names during the transformation process. It will generate the method **prediction(inputImage: CVPixelBuffer)-> modelOutPut** in Xcode project.

## Visualize CoreML Model

Visualize CoreML model with coremltools.

    python3 viz_coreml saved_models/candy.mlmodel

Visualize the model with [netron](https://github.com/lutzroeder/netron).

## Reduce the Size of CoreML Model

Use the function **convert_neural_network_spec_weights_to_fp16** in coremltools to convert the weights from **float32** to **float16**.

    python3 reduce_coreml_size.py saved_models/candy.mlmodel

Use the function **quantize_weights** in coremltools to convert the weights which uses less than **8 bits** to represent a floating point number.

    python3 quantumize_coreml_size.py saved_models/candy.mlmodel

## Referance
1. [PyTorch models](https://github.com/pytorch/examples)

2. [Building a Neural Style Transfer app on iOS with PyTorch and CoreML](https://medium.com/@alexiscreuzot/building-a-neural-style-transfer-app-on-ios-with-pytorch-and-coreml-76e00cd14b28)

3. [Quantumizing CoreML model](https://apple.github.io/coremltools/generated/coremltools.models.neural_network.quantization_utils.html)