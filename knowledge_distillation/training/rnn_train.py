#!/usr/bin/python

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import concatenate
from keras import losses
from keras import regularizers
from keras.constraints import min_max_norm
import h5py

from keras.constraints import Constraint
from keras import backend as K
import numpy as np

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.42
#set_session(tf.Session(config=config))


def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.mean(
        mymask(y_true) * (
            10 * K.square(K.sqrt(K.abs(y_pred)) - K.sqrt(K.abs(y_true))) +
            K.square(K.sqrt(K.abs(y_pred)) - K.sqrt(K.abs(y_true))) +
            0.01 * K.binary_crossentropy(y_pred, y_true)
        ),
        axis=-1
    )

#
# def mycost(y_true, y_pred):
#     return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

reg = 0.000001
constraint = WeightClip(0.499)

print('Build model...')
main_input = Input(shape=(None, 42), name='main_input')
tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
vad_gru = GRU(24, activation='tanh', reset_after=False,recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)
noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
noise_gru = GRU(48, activation='relu', reset_after=False, recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input)
denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])

denoise_gru = GRU(96, activation='tanh', reset_after=False, recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)

denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

model_teacher1 = Model(inputs=main_input, outputs=[denoise_output, vad_output])
model_teacher2 = Model(inputs=main_input, outputs=[denoise_output, vad_output])
model_student1 = Model(inputs=main_input, outputs=[denoise_output, vad_output])

model_teacher1.load_weights('conjoined.hdf5')
model_teacher2.load_weights('hogwash_echo_14sep23.hdf5')

print("Done Loading teacher model hogwash and conjoined")

model_teacher1.compile(loss=[mycost, my_crossentropy],
              metrics=[msse],
              optimizer='adam', loss_weights=[10, 0.5])

model_teacher2.compile(loss=[mycost, my_crossentropy],
              metrics=[msse],
              optimizer='adam', loss_weights=[10, 0.5])

model_student1.compile(loss=[mycost, my_crossentropy],
              metrics=[msse],
              optimizer='adam', loss_weights=[10, 0.5])

batch_size = 32

print('Loading data...')
with h5py.File('training.h5', 'r') as hf:
    all_data = hf['data'][:]
print('done.')

window_size = 2000

nb_sequences = len(all_data)//window_size
print(nb_sequences, ' sequences')
x_train = all_data[:nb_sequences*window_size, :42]
x_train = np.reshape(x_train, (nb_sequences, window_size, 42))

y_train = np.copy(all_data[:nb_sequences*window_size, 42:64])
y_train = np.reshape(y_train, (nb_sequences, window_size, 22))

noise_train = np.copy(all_data[:nb_sequences*window_size, 64:86])
noise_train = np.reshape(noise_train, (nb_sequences, window_size, 22))

vad_train = np.copy(all_data[:nb_sequences*window_size, 86:87])
vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

all_data = 0;
#x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')

print(len(x_train), 'train sequences. x shape =', x_train.shape, 'y shape = ', y_train.shape)

print('Train...')
epochs = 100
alpha = 0.5

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    for batch in range(0, len(x_train), batch_size):
        batch_inputs = x_train[batch:batch + batch_size]
        batch_targets = [y_train[batch:batch + batch_size], vad_train[batch:batch + batch_size]]

        # Obtain teacher predictions for the current batch
        teacher_predictions1 = model_teacher1.predict(batch_inputs)
        teacher_predictions2 = model_teacher2.predict(batch_inputs)

        # Train the student model with the same data and targets as in model.fit
        student_loss = model_student1.train_on_batch(batch_inputs, batch_targets)

        # Calculate distillation loss using teacher predictions and student's predictions
        distillation_loss1 = mycost(teacher_predictions1[0], model_student1.predict(batch_inputs)[0])
        distillation_loss1 += my_crossentropy(teacher_predictions1[1], model_student1.predict(batch_inputs)[1])

        distillation_loss2 = mycost(teacher_predictions2[0], model_student1.predict(batch_inputs)[0])
        distillation_loss2 += my_crossentropy(teacher_predictions2[1], model_student1.predict(batch_inputs)[1])

        # Combine the distillation losses from both teachers and cross-entropy terms
        distillation_loss = 0.5 * (distillation_loss1 + distillation_loss2)

        # Extract the individual loss values from the list
        student_loss_value = student_loss[0]
        distillation_loss_value = np.mean(distillation_loss)

        # Print shapes for debugging
        print(f"Student Loss Value: {student_loss_value}, Distillation Loss Value: {distillation_loss_value}")

        # Calculate total loss
        total_loss = student_loss_value + alpha * distillation_loss_value

        print(f"Batch Loss: {total_loss}")


# model_student1.fit(x_train, [y_train, vad_train],
#           batch_size=batch_size,
#           epochs=120,
#           validation_split=0.1)
model_student1.save("hogwash_and_conjoined.hdf5")
