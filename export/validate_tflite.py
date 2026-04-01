import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

inp  = interpreter.get_input_details()[0]
outp = interpreter.get_output_details()[0]

print("입력 shape:", inp['shape'])   # [1, 320, 320, 3] 확인
print("입력 dtype:", inp['dtype'])   # uint8 (INT8 양자화 시)
print("출력 shape:", outp['shape'])  # [1, N, 85] 등 확인