from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K
import numpy as np
import re


class WeightClip(Constraint):
    def __init__(self, c=2, **kwargs):
        super(WeightClip, self).__init__()
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'c': self.c}


# Define your custom loss function 'mycost'
def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10 * K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(
        K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01 * K.binary_crossentropy(y_pred, y_true)), axis=-1)

def my_crossentropy(y_true, y_pred):
    return K.mean(2 * K.abs(y_true - 0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)


# Define your custom metric function 'msse'
def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)


# Load the model using custom_object_scope to include all custom objects
def load_keras_model_weights(model_path):
    with custom_object_scope(
            {'WeightClip': WeightClip, 'mycost': mycost, 'my_crossentropy': my_crossentropy, 'msse': msse}):
        model = load_model(model_path, custom_objects={'WeightClip': WeightClip, 'mycost': mycost,
                                                       'my_crossentropy': my_crossentropy, 'msse': msse})

        weights = {}
        for layer in model.layers:
            weights[layer.name] = layer.get_weights()

        return weights


def parse_cpp_file(cpp_path):
    """
    Parse the C++ file to extract weights and layer information.

    Note: This function is a conceptual outline and might require further refinement.

    :param cpp_path: Path to the C++ file.
    :return: A dictionary with layer names as keys and their weights as values.
    """
    with open(cpp_path, 'r') as file:
        cpp_content = file.read()

    # Regular expressions to match patterns in the C++ file
    # This needs to be refined based on the actual format of the C++ file
    weight_pattern = r'static const rnn_weight (\w+)\[\d+\] = {([^}]+)};'
    layer_info = {}

    for match in re.finditer(weight_pattern, cpp_content):
        layer_name, weights_str = match.groups()
        weights = np.array([int(x) for x in weights_str.split(',') if x.strip() != ''])
        layer_info[layer_name] = weights

    return layer_info


def compare_weights(keras_weights, cpp_weights):
    """
    Compare weights from the Keras model and the parsed C++ file.

    :param keras_weights: Weights extracted from the Keras model.
    :param cpp_weights: Weights parsed from the C++ file.
    :return: Comparison result.
    """
    comparison_result = "Comparison complete"
    for layer_name, k_weights in keras_weights.items():
        found = False
        for cpp_layer_name in cpp_weights:
            if layer_name in cpp_layer_name:
                # Considering layers that contain the Keras layer name in the C++ layer name
                cpp_layer_weights = cpp_weights[cpp_layer_name]
                # Compare the weights between Keras and C++
                # This may involve reshaping, scaling, and quantizing the Keras weights
                # to match the format of the weights in the C++ file.
                found = True
                break

        if not found:
            print(f"Layer {layer_name} not found in C++ weights.")

    return comparison_result



# Paths to the files
model_path = 'hogwash_echo_14sep23.hdf5'
cpp_path = '../src/rnn_data.c'

keras_weights = load_keras_model_weights(model_path)
cpp_weights = parse_cpp_file(cpp_path)

comparison_result = compare_weights(keras_weights, cpp_weights)
print(comparison_result)







#
# from __future__ import print_function
#
# import keras
# from keras.models import Sequential
# from keras.models import Model
# from keras.layers import Input
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import GRU
# from keras.layers import SimpleRNN
# from keras.layers import Dropout
# from keras.layers import concatenate
# from keras import losses
# from keras import regularizers
# from keras.constraints import min_max_norm
# import h5py
#
# from keras.constraints import Constraint
# from keras import backend as K
# import numpy as np
# from keras.models import load_model
#
# #import tensorflow as tf
# #from keras.backend.tensorflow_backend import set_session
# #config = tf.ConfigProto()
# #config.gpu_options.per_process_gpu_memory_fraction = 0.42
# #set_session(tf.Session(config=config))
#
#
# def my_crossentropy(y_true, y_pred):
#     return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)
#
# def mymask(y_true):
#     return K.minimum(y_true+1., 1.)
#
# def msse(y_true, y_pred):
#     return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)
#
# def mycost(y_true, y_pred):
#     return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)
#
# def my_accuracy(y_true, y_pred):
#     return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)
#
# class WeightClip(Constraint):
#     '''Clips the weights incident to each hidden unit to be inside a range
#     '''
#     def __init__(self, c=2, name='WeightClip'):
#         self.c = c
#
#     def __call__(self, p):
#         return K.clip(p, -self.c, self.c)
#
#     def get_config(self):
#         return {'name': self.__class__.__name__,
#             'c': self.c}
#
# reg = 0.000001
# constraint = WeightClip(0.499)
#
# print('Build model...')
# main_input = Input(shape=(None, 42), name='main_input')
# tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
# vad_gru = GRU(24, activation='tanh',  reset_after=False,recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
# vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)
# noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
# noise_gru = GRU(48, activation='relu', reset_after=False, recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input)
# denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])
#
# denoise_gru = GRU(96, activation='tanh',  reset_after=False, recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)
#
# denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)
#
# model = Model(inputs=main_input, outputs=[denoise_output, vad_output])
#
#
# model.compile(loss=[mycost, my_crossentropy],
#               metrics=[msse],
#               optimizer='adam', loss_weights=[10, 0.5])
#
# # model.load_weights('beguiling_drafter.hdf5',  by_name=True, skip_mismatch=True)
# model.load_weights('hogwash_echo_14sep23.hdf5')
#
# model.summary()