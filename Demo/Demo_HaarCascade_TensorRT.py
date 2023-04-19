from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.applications.vgg19 import VGG19
import numpy as np
import os

# Load model =>Predict => save model
model = ResNet50(weights='imagenet')
model.save('/content/resnet50_saved_model')

# Convert to TF_TRT => SavedModel
print('Converting to TF-TRT FP32 or FP16 or INT8...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16,
    max_workspace_size_bytes=8000000000
)
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='resnet50_saved_model',
    conversion_params=conversion_params
)

# Converter method used to partition and optimize TensorRT compatible segments
converter.convert()

# Save the model to the disk 
converter.save(output_saved_model_dir='resnet50_saved_model_TFTRT_FP32')
print('Done Converting to TF-TRT FP32')
