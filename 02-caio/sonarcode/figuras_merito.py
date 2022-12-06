import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
import matplotlib.pyplot as plt


# indice sp
def sp_index(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    recall = recall_score(y_true, y_pred, average=None)
    sp = np.sqrt(np.sum(recall) / num_classes *
                 np.power(np.prod(recall), 1.0 / float(num_classes)))
    return sp
  

# plotar matriz confusão
def plota_confusao(model,x, y):
  y_pred = model.predict_classes(x).argmax(axis=-1)
  fig, ax = plt.subplots()
  confusao = confusion_matrix(y, y_pred)
  sns.heatmap(confusao, annot=True, 
            ax=ax, fmt='d', cmap='Reds')
  ax.set_title("Matriz de Confusão", fontsize=18)
  ax.set_ylabel("True label")
  ax.set_xlabel("Predicted Label")
  plt.tight_layout()
  

# Curva de validação
def curvalearnvalid(train_history, path):
  # list all data in history
  # print(train_history.history.keys())
  # summarize history for accuracy
  plt.plot(train_history.history['accuracy'])
  plt.plot(train_history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'valid'], loc='upper left')
  plt.savefig(str(path)+'_acc.png')  
  plt.close()
  # summarize history for loss
  plt.plot(train_history.history['loss'])
  plt.plot(train_history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'valid'], loc='upper left')
  plt.savefig(str(path)+'_loss.png')
  plt.close()
  
  
def sitrep(acc_fold, loss_fold, val_fold, sp_fold, best_model, scores, sp, out):
  print('------------------------------------------------------------------------')
  print('Resultado dos folds')
  for i in range(0, len(acc_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_fold[i]} - Accuracy: {acc_fold[i]}% - Validação: {val_fold[i]}% - sp: {sp_fold[best_model]}%')
  print('------------------------------------------------------------------------')
  print('Melhor modelo:')
  print(f'> Fold {best_model+1} - Loss: {loss_fold[best_model]} - Accuracy: {acc_fold[best_model]}% - Validação: {val_fold[best_model]}% - sp: {sp_fold[best_model]}%')
  print('------------------------------------------------------------------------')
  print('Média dos folds:')
  print(f'> Accuracy: {np.mean(val_fold)} (+- {np.std(val_fold)})')
  print(f'> Loss: {np.mean(loss_fold)}')
  print('------------------------------------------------------------------------')
  print('Média dos Testes:')
  print(f'> Loss: {scores[0]} - Accuracy: {scores[1] * 100}%')
  print('------------------------------------------------------------------------')
  print('indice sp: ', np.mean(sp),'%')
  print('------------------------------------------------------------------------')
  print(out)
  print('------------------------------------------------------------------------')
