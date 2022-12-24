import tensorflow as tf

import GPU_MEMORY


def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print('gpus true')
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_MEMORY.MAX_MEM)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
