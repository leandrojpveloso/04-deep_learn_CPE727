import numpy as np 

def generate_images(run_data, window_size: int, window_step: int, trgt_multiplier: int, normalize = True):
    pruned_indexes = list(range(0, len(run_data) - window_size, window_step))
    
    # Creating a tensor with dimensions #of_images x window size x spectrum bins
    data_shape = (
        len(pruned_indexes), window_size, run_data.shape[1], 1)
    # Allocating memory for the tensors
    image_data = np.zeros(shape=data_shape)

    # Separating the data
    for image_index, spectrum_index in enumerate(pruned_indexes):
        windowed_data = run_data[spectrum_index:spectrum_index + window_size, :]
        if normalize:
            windowed_data = windowed_data / np.sqrt(np.sum(windowed_data**2, axis=1, keepdims=True))
        windowed_data = np.array(windowed_data.reshape(windowed_data.shape[0], 
                                                       windowed_data.shape[1], 1), np.float64)
        image_data[image_index] = windowed_data
    return image_data


def generate_datalofar_trgt(processed_data_dict, tipo, sublofar_config, trgt_label_map=None):

    window_size = sublofar_config.sublofar_size
    window_step = sublofar_config.sublofar_step
    
    if trgt_label_map is None:
        trgt_label_map = classe_target
    
    if tipo == "lofar":
  
      trgt = np.concatenate(
          [trgt_label_map[cls_name]*np.ones(Sxx[0].shape[0]) 
          for cls_name, run in processed_data_dict.items() 
          for run_name, Sxx in run.items()]
          )
      data = np.concatenate(
          [Sxx[0]
          for cls_name, run in processed_data_dict.items() 
          for run_name, Sxx in run.items()], axis=0
          )
      
    elif tipo == "sublofar":  
      data = np.concatenate(
          [generate_images(Sxx[0],window_size=window_size,
                          window_step=window_step, 
                          trgt_multiplier=trgt_label_map[cls_name])
          for cls_name, run in processed_data_dict.items() 
          for run_name, Sxx in run.items()], axis=0
          )
      trgt = np.concatenate(
          [trgt_label_map[cls_name]*np.ones(
              ((generate_images(Sxx[0],window_size=window_size,
                          window_step=window_step, 
                          trgt_multiplier=trgt_label_map[cls_name])
              ).shape[0],1)
                                            ) 
          for cls_name, run in processed_data_dict.items() 
          for run_name, Sxx in run.items()]
          )
    return data, trgt
