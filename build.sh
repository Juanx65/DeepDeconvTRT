#!/bin/bash

# remeber to do a chmod +x build.sh befor runing with ./build.sh
BATCH_SIZE=$1 #128

C=1
H=24
W=1024

echo $INPUT_SHAPE

#TRT FP32
python onnx_transform.py --weights="weights/best_fp32.pth"  --input_shape $BATCH_SIZE $C $H $W
python build_trt.py --weights="weights/best_fp32.onnx"  --fp32 --input_shape $BATCH_SIZE $C $H $W

#TRT FP16
python onnx_transform.py --weights="weights/best_fp16.pth" --input_shape $BATCH_SIZE $C $H $W
python build_trt.py --weights="weights/best_fp16.onnx"  --fp16 --input_shape $BATCH_SIZE $C $H $W

#TRT INT8
#rm -r outputs/cache
#python onnx_transform.py --weights="weights/best_int8.pth" --input_shape $BATCH_SIZE $C $H $W
#python build_trt.py --weights="weights/best_int8.onnx"  --int8 --input_shape $BATCH_SIZE $C $H $W