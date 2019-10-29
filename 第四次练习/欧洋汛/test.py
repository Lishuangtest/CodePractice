import tensorflow as tf
print(
tf.test.is_built_with_cuda(),
tf.test.gpu_device_name(),
tf.test.is_gpu_available())
