# TensorRT PoseNet

## Description
This sample contains code that convert TensorFlow Lite PoseNet model to ONNX model and performs TensorRT inference.
1. Download TensorFlow Lite PoseNet Model.
2. Convert to ONNX Model.
3. Convert ONNX Model to Serialize engine and inference .

## Environment
- CUDA 12.1 or 11.4
- Deepstream 6.3 or 6.0.1 

## Convert ONNX Model on your Host PC

### Download TensorFlow Lite model
Download PoseNet's TensorFlow Lite Model from the For TensorFlow Hub.  
`posenet_mobilenet_float_075_1_default_1.tflite`
- https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1


### Convert ONNX Model

Install onnxruntime and tf2onnx.
```
pip3 install onnxruntime tf2onnx
```

Convert TensorFlow Lite Model to ONNX Model.  
```
python3 -m tf2onnx.convert --opset 13 \
    --tflite ./posenet_mobilenet_float_075_1_default_1.tflite \
    --output ./posenet_mobilenet_float_075_1_default_1.onnx
```

## Run 

The following is executed on dGpu.

### Install dependency
Install pycuda.  
See details:
```
sudo apt update
sudo apt install python3-dev
pip3 install numpy==1.23.1
pip3 install --user cython
pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda
pip install opencv-python
pip install tensorrt==8.5.3.1 
```

### Clone this repository.
Clone repository.
```
cd ~
git clone https://github.com/GOBINATH1992/tensorrt-examples
cd posenet
git submodule update --init --recursive
```

### Convert to Serialize engine file.
Copy `posenet_mobilenet_float_075_1_default_1.onnx` and check model.
```
/usr/src/tensorrt/bin/trtexec --onnx=./posenet_mobilenet_float_075_1_default_1.onnx --saveEngine=posenet_mobilenet_float_075_1_default_1.trt
```

If you want to convert to FP16 model, add --fp16 to the argument of convert_onnxgs2trt.py
``
python3 convert_onnxgs2trt.py \
    --model /posenet_mobilenet_float_075_1_default_1.onnx \
    --output posenet_mobilenet_float_075_1_default_1_fp16.trt \
    --fp16
```


Finally you can run the demo.
```
python3 trt_simgle_posenet.py \
    --model posenet_mobilenet_float_075_1_default_1.trt  --output out.mp4 --videopath /opt/nvidia/deepstream/deepstream-6.3/samples/streams/sample_ride_bike.mov
```

