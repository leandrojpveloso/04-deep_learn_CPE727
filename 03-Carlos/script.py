import numpy as np
import keras
from keras import layers
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from scikeras.wrappers import KerasRegressor

file_path = '/home/carlos.dias/carlos.dias/data/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et4_eta0.npz'

leblon_data = np.load(file_path)

random_seed = 42
output_file = '/home/carlos.dias/carlos.dias/results.npy'

ring_data = leblon_data['data'][:, 1:101]
X_false = ring_data[leblon_data['target'] == 0.0]
X_true = ring_data[leblon_data['target'] == 1.0]

X_train, X_test = train_test_split(X_false, test_size=0.3, random_state=random_seed)

def build_model(model_layers, input_dim=100, output_layer=(100, 'linear'), optimizer='adam', loss='mean_squared_error', batch_norm=False):
    input_layer = keras.Input(shape=(input_dim,))
    layer = input_layer
    for dim, activation in model_layers:
        layer = layers.Dense(dim, activation=activation)(layer)
        if batch_norm:
            layer = layers.BatchNormalization()(layer)
    layer = layers.Dense(output_layer[0], activation=output_layer[1])(layer)
    model = keras.Model(input_layer, layer)
    model.compile(optimizer=optimizer, loss=loss)
    return model

outer_loop = 2
inner_loop = 2
batch_size = 256
epochs = 10
# parameters
model_1 = [
    (50, 'gelu'),
    (5, 'gelu'),
    (50, 'gelu'),
]
model_2 = [
    (50, 'relu'),
    (10, 'relu'),
    (50, 'relu'),
]
# full grid
param_grid = {
    'model__model_layers': [model_1, model_2],
    'model__batch_norm': [True, False],
}

results = []
k = 0
cv_outer = KFold(n_splits=outer_loop, shuffle=True, random_state=1)

for train_ix, test_ix in cv_outer.split(X_train):
    X_kfold_train, X_kfold_test = X_train[train_ix, :], X_train[test_ix, :]
    # inner cross-validation
    cv_inner = KFold(n_splits=inner_loop, shuffle=True, random_state=random_seed)
    # model
    model = KerasRegressor(build_model, epochs=epochs, batch_size=batch_size, verbose=1)
    # search
    search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', n_jobs=1, cv=cv_inner, refit=True)
    # execute search
    result = search.fit(X_kfold_train, X_kfold_train)
    # evaluate the model
    best_model = result.best_estimator_
    X_test_eval = best_model.predict(X_test)
    mse = mean_squared_error(X_test, X_test_eval)
    # store the result
    result_k = {
        'results': result.cv_results_,
        'outer_loop_mse': mse,
    }
    results.append(result_k)
    # save scores
    with open(output_file, 'wb') as output:
        np.save(output, results)
    k += 1
    # report progress
    print('>mse=%.3f, est=%.3f, cfg=%s' % (mse, result.best_score_, result.best_params_))
