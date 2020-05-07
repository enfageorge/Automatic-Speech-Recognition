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


def CnnRnnModel(inputDim, filters, kernelSize, convStride,
                convBorderMode, units, outputDim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    inputData = Input(name='the_input', shape=(None, inputDim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernelSize,
                     strides=convStride,
                     padding=convBorderMode,
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
        x, kernelSize, convBorderMode, convStride)
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


def finalModel(inputDim, filters, kernelSize, convStride,
               convBorderMode, units, outputDim=29, dropoutRate=0.5, numberOfLayers=2,
               cell=GRU, activation='tanh'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    inputData = Input(name='the_input', shape=(None, inputDim))
    # TODO: Specify the layers in your network
    conv_1d = Conv1D(filters, kernelSize,
                     strides=convStride,
                     padding=convBorderMode,
                     activation='relu',
                     name='layer_1_conv',
                     dilation_rate=1)(inputData)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)

    if numberOfLayers == 1:
        layer = Bidirectional(cell(units, activation=activation,
                                   return_sequences=True, implementation=2, name='rnn_1', dropout=dropoutRate))(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = Bidirectional(cell(units, activation=activation,
                                   return_sequences=True, implementation=2, name='rnn_1', dropout=dropoutRate))(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(numberOfLayers - 2):
            layer = Bidirectional(cell(units, activation=activation,
                                       return_sequences=True, implementation=2, name='rnn_{}'.format(i + 2), dropout=dropoutRate))(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(i + 2))(layer)

        layer = Bidirectional(cell(units, activation=activation,
                                   return_sequences=True, implementation=2, name='final_layer_of_rnn'))(layer)
        layer = BatchNormalization(name='bt_rnn_final')(layer)

    time_dense = TimeDistributed(Dense(outputDim))(layer)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=inputData, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernelSize, convBorderMode, convStride)
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn =  Bidirectional(GRU(units, return_sequences=True,
                                   implementation=2, name='rnn'), 
                               merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model




def finalModel(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, dropout_rate=0.5, number_of_layers=2, 
    cell=GRU, activation='tanh'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='layer_1_conv',
                     dilation_rate=1)(input_data)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)


    if number_of_layers == 1:
        layer = Bidirectional(cell(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate))(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = Bidirectional(cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate))(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(number_of_layers - 2):
            layer = Bidirectional(cell(units, activation=activation,
                        return_sequences=True, implementation=2, name='rnn_{}'.format(i+2), dropout=dropout_rate))(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(i+2))(layer)

        layer = Bidirectional(cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='final_layer_of_rnn'))(layer)
        layer = BatchNormalization(name='bt_rnn_final')(layer)
    

    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model 
