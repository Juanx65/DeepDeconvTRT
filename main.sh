#!/bin/bash

# remeber to do a chmod +x main.sh befor runing with ./main.sh
BATCH_SIZE=$1

# remeber to do a chmod +x build.sh befor runing with ./build.sh
#./build.sh "$BATCH_SIZE"  > /dev/null 2>&1 # cambiar  "$BATCH_SIZE" por -1 para construir la red de forma dinamica, no aplica a int8

#VANILLA
python main.py --network vanilla --weights weights/best.pth --numpy_data

#TRT FP32
python main.py --network trtfp32 -trt --engine='weights/best_fp32.engine' --numpy_data

#TRT FP16
python main.py --network trtfp16 -trt --engine='weights/best_fp16.engine' --numpy_data