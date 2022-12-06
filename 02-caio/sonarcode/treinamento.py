import numpy as np
from sklearn import preprocessing
import pandas as pd
import datetime
import keras.callbacks as callbacks
from sklearn.model_selection import StratifiedKFold, KFold

# função de padronização
def padronizaconjunto(data, tipo):
  #cria o objeto obj_std
  if tipo == 'mapstd':
    obj_std = preprocessing.StandardScaler().fit(data)
  elif tipo == 'mapstd_rob':
    obj_std = preprocessing.RobustScaler().fit(data)  
  elif tipo == 'mapminmax':
    obj_std = preprocessing.MinMaxScaler().fit(data)
  data_std = obj_std.transform(data) #aplica o padronizador nos dados
  data_std = pd.DataFrame(data_std) # transforma em dataFrame
  return data_std, obj_std


# função para formatar os dados de entrada para a rede cnn
def format_inputlstm(data):
  data_array = np.array(data[:])
  x_out = data_array.reshape(data_array.shape[0],data_array.shape[1],1)
  return x_out


# Difinindo quem é o melhor modelo
def melhor_modelo(sp_fold,val_fold):  
  n_max = sp_fold.count(max(sp_fold))
  if n_max > 1:
    ind_model = [(n) for n, x in enumerate(sp_fold) if x== max(sp_fold)] # Quais indices?
    val_model = [val_fold[x] for x in ind_model] # Valores de validação dos indices
    return val_model.index(max(val_model))
  else:
    return sp_fold.index(max(sp_fold))

  
# função de controle (TensorBoard + Early Stopping + CheckPoint)
def control_train(train_config, modelpath):

  #tensor_board = [callbacks.TensorBoard(log_dir=log_dir)]

  earlystopping = callbacks.EarlyStopping(monitor='val_loss',
                                          patience=train_config.patience, 
                                          verbose=True,
                                          restore_best_weights=True,
                                          mode='min')
    
  checkpoint = callbacks.ModelCheckpoint(filepath=modelpath, 
                                         monitor='val_loss',
                                         verbose=0, 
                                         save_best_only=True,
                                         mode='min')
 
  return [earlystopping,checkpoint]

# separar folds para o treinamento

def folds_treinamento(config_train, x_data, y_data):
    dict_train = dict.fromkeys(list(range(config_train.split)) , None)
    dict_valid = dict.fromkeys(list(range(config_train.split)) , None)
    div_fold, div_test = [], []
    ExtSplitKfold = StratifiedKFold(n_splits=config_train.split, 
                                shuffle=True, random_state=config_train.seed)
    n_test = 0
    for id_fold, id_test in ExtSplitKfold.split(x_data, y_data):
        div_test.append(id_test)
        div_fold.append(id_fold)
        n_fold = 0
        div_train, div_valid = [], []
        # K-fold Cross Validation model evaluation
        IntSplitKfold = StratifiedKFold(n_splits= config_train.folds, 
                                        shuffle=True, random_state=config_train.seed)
        for id_train, id_valid in IntSplitKfold.split(x_data[id_fold], y_data[id_fold]):
            div_train.append(id_train)
            div_valid.append(id_valid)
            n_fold = n_fold+1
        dict_train[n_test] = div_train
        dict_valid[n_test] = div_valid
        n_test = n_test+1
    return div_fold, div_test, dict_train, dict_valid

def folds_treinamento_corrida(config_train, x_data):
    dict_train = dict.fromkeys(list(range(config_train.split)) , None)
    dict_valid = dict.fromkeys(list(range(config_train.split)) , None)
    div_fold, div_test = [], []
    ExtSplitKfold = KFold(n_splits=config_train.split, 
                                shuffle=True, random_state=config_train.seed)
    n_test = 0
    for id_fold, id_test in ExtSplitKfold.split(x_data):
        div_test.append(id_test)
        div_fold.append(id_fold)
        n_fold = 0
        div_train, div_valid = [], []
        # K-fold Cross Validation model evaluation
        IntSplitKfold = KFold(n_splits= config_train.folds, 
                                        shuffle=True, random_state=config_train.seed)
        for id_train, id_valid in IntSplitKfold.split(x_data[id_fold]):
            div_train.append(id_train)
            div_valid.append(id_valid)
            n_fold = n_fold+1
        dict_train[n_test] = div_train
        dict_valid[n_test] = div_valid
        n_test = n_test+1
    return div_fold, div_test, dict_train, dict_valid
  
