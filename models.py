from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)


def simpleRnnModel(inputDim, outputDim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    inputData = Input(name='the_input', shape=(None, inputDim))
    # Add recurrent layer
    simpleRNN = GRU(outputDim, return_sequences=True,
                    implementation=2, name='rnn')(inputData)
    # Add softmax activation layer
    predictY = Activation('softmax', name='softmax')(simpleRNN)
    # Specify the model
    model = Model(inputs=inputData, outputs=predictY)
    model.outputLength = lambda x: x
    print(model.summary())
    return model


def RNNModel(inputDim, units, activation, outputDim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    inputData = Input(name='the_input', shape=(None, inputDim))
    # Add recurrent layer
    simpleRNN = GRU(units, activation=activation,
                    return_sequences=True, implementation=2, name='rnn', dropout=0.3)(inputData)
    # TODO: Add batch normalization
    BnRNN = BatchNormalization()(simpleRNN)
    # TODO: Add a TimeDistributed(Dense(outputDim)) layer
    TimeDense = TimeDistributed(Dense(units=outputDim))(BnRNN)
    # Add softmax activation layer
    predictY = Activation('softmax', name='softmax')(TimeDense)
    # Specify the model
    model = Model(inputs=inputData, outputs=predictY)
    model.outputLength = lambda x: x
    print(model.summary())
    return model


def CnnRnnModel(inputDim, filters, kernel_size, conv_stride,
                conv_border_mode, units, outputDim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    inputData = Input(name='the_input', shape=(None, inputDim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(inputData)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simpleRNN = SimpleRNN(units, activation='relu', dropout=0.3,
                          return_sequences=True, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    BnRNN = BatchNormalization()(simpleRNN)
    # TODO: Add a TimeDistributed(Dense(outputDim)) layer
    TimeDense = TimeDistributed(Dense(units=outputDim))(BnRNN)
    # Add softmax activation layer
    predictY = Activation('softmax', name='softmax')(TimeDense)
    # Specify the model
    model = Model(inputs=inputData, outputs=predictY)
    model.outputLength = lambda x: cnn_outputLength(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_outputLength(input_length, filter_size, border_mode, stride,
                     dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        outputLength = input_length
    elif border_mode == 'valid':
        outputLength = input_length - dilated_filter_size + 1
    return (outputLength + stride - 1) // stride


def DeppRnnModel(inputDim, units, recur_layers, outputDim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    inputData = Input(name='the_input', shape=(None, inputDim))
    # TODO: Add recurrent layers, each with batch normalization
    BnRNN = inputData
    for i in range(recur_layers):
        layer_name = 'rnn_' + str(i)
        simpleRNN = GRU(units, activation='relu',
                        return_sequences=True, implementation=2, name=layer_name)(BnRNN)
        BnRNN = BatchNormalization()(simpleRNN)
    # TODO: Add a TimeDistributed(Dense(outputDim)) layer
    TimeDense = TimeDistributed(Dense(units=outputDim))(BnRNN)
    # Add softmax activation layer
    predictY = Activation('softmax', name='softmax')(TimeDense)
    # Specify the model
    model = Model(inputs=inputData, outputs=predictY)
    model.outputLength = lambda x: x
    print(model.summary())
    return model


def BidirectionalRnnModel(inputDim, units, outputDim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    inputData = Input(name='the_input', shape=(None, inputDim))
    # TODO: Add bidirectional recurrent layer
    BidirRnn = Bidirectional(GRU(units, return_sequences=True,
                                 implementation=2, name='rnn'),
                             merge_mode='concat')(inputData)
    # TODO: Add a TimeDistributed(Dense(outputDim)) layer
    TimeDense = TimeDistributed(Dense(units=outputDim))(bidir_rnn)
    # Add softmax activation layer
    predictY = Activation('softmax', name='softmax')(TimeDense)
    # Specify the model
    model = Model(inputs=inputData, outputs=predictY)
    model.outputLength = lambda x: x
    print(model.summary())
    return model