import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# You can also get more details about your GPU devices
for gpu in tf.config.list_physical_devices('GPU'):
    print(gpu)