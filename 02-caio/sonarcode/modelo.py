# implementando camada LSTM
import keras
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import Input, Model, Sequential
#from tensorflow.keras import Input, Model, Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, Reshape, Conv1D, Conv2D, Concatenate, Add
from keras.utils.vis_utils import plot_model


def modelolstm(n_timesteps, n_features, n_outputs, model_config):
  model = Sequential()
  model.add(LSTM(model_config.neulstm_1, input_shape=(n_timesteps,n_features)))
  if model_config.use_drop:
    model.add(Dropout(model_config.drop))
  if model_config.use_flatten:
    model.add(Flatten())
  if model_config.neumlp_1 != 0:
    model.add(Dense(model_config.neumlp_1, activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))
  model.add(Dense(n_outputs, activation=model_config.funcout))
  model.compile(optimizer= model_config.optimizer, loss= model_config.loss, metrics=[model_config.metrics])
  return model

def modelomlp(data, num_classes, model_config):
  n_steps = data.shape[1]
  model = Sequential()
  model.add(Dense(model_config.neumlp_1, activation=model_config.funcactiv, input_dim=n_steps))
  if model_config.use_drop:
    model.add(Dropout(model_config.drop))
  if model_config.neumlp_2 != 0:
    model.add(Dense(model_config.neumlp_2, activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))
  model.add(Dense(num_classes, activation=model_config.funcout))
  # opt = Adam(lr=model_config.opt_lr,beta_1=model_config.opt_beta)
  model.compile(optimizer='adam', loss=model_config.loss, metrics=[model_config.metrics])
  return model

def modelocnn(data, num_classes, model_config):
  model = Sequential()
  input_shape=(data.shape[1:])
  model.add(Conv2D(model_config.neucnn_1, kernel_size=model_config.kernel_size,padding = 'same', activation=model_config.funcactiv, input_shape=input_shape))
  if model_config.use_drop:
    model.add(Dropout(model_config.drop))
  if model_config.neucnn_2 != 0:
    model.add(Conv2D(model_config.neucnn_2, kernel_size=model_config.kernel_size,padding = 'same',activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))
  model.add(Flatten())
  if model_config.neumlp_1 != 0:
    model.add(Dense(model_config.neumlp_1, activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))  
  if model_config.neumlp_2 != 0:
    model.add(Dense(model_config.neumlp_2, activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))
  model.add(Dense(num_classes, activation=model_config.funcout))
  model.compile(optimizer=model_config.optimizer, loss=model_config.loss, metrics=[model_config.metrics])
  return model

def modelocnn1(data, num_classes, model_config):
  input_shape=(data.shape[1],1)
  model = Sequential()
  model.add(Conv1D(model_config.neucnn_1, kernel_size=model_config.kernel_size,padding = 'same', activation=model_config.funcactiv, input_shape=input_shape))
  if model_config.use_drop:
    model.add(Dropout(model_config.drop))
  if model_config.neucnn_2 != 0:
    model.add(Conv1D(model_config.neucnn_2, kernel_size=model_config.kernel_size,padding = 'same',activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))
  model.add(Flatten())
  if model_config.neumlp_1 != 0:
    model.add(Dense(model_config.neumlp_1, activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))  
  if model_config.neumlp_2 != 0:
    model.add(Dense(model_config.neumlp_2, activation=model_config.funcactiv))
    if model_config.use_drop:
      model.add(Dropout(model_config.drop))
  model.add(Dense(num_classes, activation=model_config.funcout))
  model.compile(optimizer=model_config.optimizer, loss=model_config.loss, metrics=[model_config.metrics])
  return model

def modelosublstm(data, n_outputs, model_config):

  input_shapedata = (data.shape[1:])
  in_subdata = Input(shape=input_shapedata)

  X = Reshape((input_shapedata[0], input_shapedata[1]), input_shape=input_shapedata)(in_subdata)
  X = LSTM(model_config.neulstm_1)(X)
  if model_config.use_drop:
    X = Dropout(model_config.drop)(X)
  X = Flatten()(X)
  if model_config.neumlp_1 != 0:
    X = Dense(model_config.neumlp_1, activation=model_config.funcactiv)(X)
  output = Dense(n_outputs, activation=model_config.funcout)(X)

  return Model(in_subdata, output)

def modelo2sublstm(data_lofar, data_tempo, n_outputs, model_config, model_config2):

  input_shapelofar = (data_lofar.shape[1:])
  input_shapetempo = (data_tempo.shape[1:])

  in_sublofar = Input(shape=input_shapelofar)
  in_subtempo = Input(shape=input_shapetempo)

  X = Reshape((input_shapelofar[0], input_shapelofar[1]), input_shape=input_shapelofar)(in_sublofar)
  X = LSTM(model_config.neulstm_1)(X)
  if model_config.use_drop:
    X = Dropout(model_config.drop)(X)
  X = Flatten()(X)
  if model_config.neumlp_1 != 0:
    X = Dense(model_config.neumlp_1, activation=model_config.funcactiv)(X)
  outputlofar = Dense(n_outputs, activation=model_config.funcout)(X)

  # _______________________________________________________________________

  W = Reshape((input_shapetempo[0], input_shapetempo[1]), input_shape=input_shapetempo)(in_subtempo)
  W = LSTM(model_config2.neulstm_1)(W)
  if model_config2.use_drop:
    W = Dropout(model_config2.drop)(W)
  W = Flatten()(W)
  if model_config2.neumlp_1 != 0:
    W = Dense(model_config2.neumlp_1, activation=model_config2.funcactiv)(W)
  outputtempo = Dense(n_outputs, activation=model_config2.funcout)(W)

  # merge imagem gen e label input
  merge=Concatenate()([outputlofar,outputtempo])
  output = Dense(n_outputs, activation=model_config.funcout)(merge)

  return Model([in_sublofar, in_subtempo], output)

def modelo2lstm(data_lofar, data_tempo, n_outputs, model_config, model_config2):

  shape_lofar = (data_lofar.shape[1],1)
  shape_tempo = (data_tempo.shape[1],1)

  in_lofar = Input(shape=shape_lofar)
  in_tempo = Input(shape=shape_tempo)

  X = LSTM(model_config.neulstm_1)(in_lofar)
  if model_config.use_drop:
    X = Dropout(model_config.drop)(X)
  X = Flatten()(X)
  if model_config.neumlp_1 != 0:
    X = Dense(model_config.neumlp_1, activation=model_config.funcactiv)(X)
  outputlofar = Dense(n_outputs, activation=model_config.funcout)(X)
  # _______________________________________________________________________
  W = LSTM(model_config2.neulstm_1)(in_tempo)
  if model_config2.use_drop:
    W = Dropout(model_config2.drop)(W)
  W = Flatten()(W)
  if model_config2.neumlp_1 != 0:
    W = Dense(model_config2.neumlp_1, activation=model_config2.funcactiv)(W)
  outputtempo = Dense(n_outputs, activation=model_config2.funcout)(W)
  # merge imagem gen e label input
  merge=Concatenate()([outputlofar,outputtempo])
  output = Dense(n_outputs, activation=model_config.funcout)(merge)
  return Model([in_lofar, in_tempo], output)


def modelo2mlp(data_lofar, data_tempo, n_outputs, model_config, model_config2):

  shape_lofar = data_lofar.shape[1]
  shape_tempo = data_tempo.shape[1]

  in_lofar = Input(shape=shape_lofar)
  in_tempo = Input(shape=shape_tempo)

  X = Dense(model_config.neumlp_1)(in_lofar)
  if model_config.use_drop:
    X = Dropout(model_config.drop)(X)
  if model_config.neumlp_2 != 0:
    X = Dense(model_config.neumlp_2, activation=model_config.funcactiv)(X)
  outputlofar = Dense(n_outputs, activation=model_config.funcactiv)(X)
  # _______________________________________________________________________
  W = Dense(model_config2.neumlp_1)(in_tempo)
  if model_config2.use_drop:
    W = Dropout(model_config2.drop)(W)
  if model_config2.neumlp_2 != 0:
    W = Dense(model_config2.neumlp_2, activation=model_config2.funcactiv)(W)
  outputtempo = Dense(n_outputs, activation=model_config2.funcactiv)(W)
  
  # merge imagem gen e label input
  merge=Concatenate()([outputlofar,outputtempo])
  model_expert = Dense(2*n_outputs, activation=model_config.funcactiv)(merge)
  output = Dense(n_outputs, activation=model_config.funcout)(model_expert)
  
  return Model([in_lofar, in_tempo], output)



def modelomlplstm(data_lofar, data_tempo, n_outputs, model_config, model_config2):

  shape_lofar = data_lofar.shape[1]
  shape_tempo = (data_tempo.shape[1],1)

  in_lofar = Input(shape=shape_lofar)
  in_tempo = Input(shape=shape_tempo)

  X = Dense(model_config.neumlp_1)(in_lofar)
  if model_config.use_drop:
    X = Dropout(model_config.drop)(X)
  if model_config.neumlp_2 != 0:
    X = Dense(model_config.neumlp_2, activation=model_config.funcactiv)(X)
  outputlofar = Dense(n_outputs, activation=model_config.funcactiv)(X)
  # _______________________________________________________________________
  W = LSTM(model_config2.neulstm_1)(in_tempo)
  if model_config2.use_drop:
    W = Dropout(model_config2.drop)(W)
  W = Flatten()(W)
  if model_config2.neumlp_1 != 0:
    W = Dense(model_config2.neumlp_1, activation=model_config2.funcactiv)(W)
  outputtempo = Dense(n_outputs, activation=model_config2.funcactiv)(W)
  
  # merge imagem gen e label input
  merge=Concatenate()([outputlofar,outputtempo])
  model_expert = Dense(2*n_outputs, activation=model_config.funcactiv)(merge)
  output = Dense(n_outputs, activation=model_config.funcout)(model_expert)
  
  return Model([in_lofar, in_tempo], output)
  
