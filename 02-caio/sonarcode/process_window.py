# Função rolling winddow
# https://gist.github.com/seberg/3866040

import numpy as np

def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.
    
    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.       
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.
    
    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.
    
    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]]],
           [[[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]]])
    
    Or to create non-overlapping windows, but only along the first dimension:
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])
    
    Note that the 0 is discared, so that the output dimension is 3:
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)
    
    This is useful for example to calculate the maximum in all (overlapping)
    2x2 submatrixes:
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])
           
    Or delay embedding (3D embedding with delay 2):
    >>> x = np.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int) # maybe crude to cast to int...
    
    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w
    
    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.") 

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps
        
        if np.any(asteps < 1):
             raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps
    
    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
             raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1 # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape # just renaming...
    
    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window==0] = 1
    
    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape
    
    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps
    
    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _
        
        new_shape = np.zeros(len(shape)*2, dtype=int)
        new_strides = np.zeros(len(shape)*2, dtype=int)
        
        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides
    
    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]
    
    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def generate_subtempo(run_data, window_size: int, window_step: int, trgt_multiplier: int, normalize = True):
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

def separar_spectro(data_dict, tipo, subtempo_config, _tam, _step, label_tipo, trgt = None):
    """
    Auxiliary function that generates a data-target pair from the sonar runs.

    params:
        data_dict (SonarDict):
            nested dicionary in which the basic unit contains
            a record of the audio (signal key) in np.array format and the
            sample_rate (fs key) stored in floating point.
        data_dict (dict):
            dictionary with target labels.
    returns:
        (np.array, np.array): Returns a tuple of data/target numpy arrays.
    """
    window_size = subtempo_config.subtempo_size
    window_step = subtempo_config.subtempo_step
    
    if trgt is None:
      trgt = trgt
    
    if tipo == "tempo":
        label = np.concatenate(
            [trgt[cls_name]*np.ones((rolling_window(dados,_tam, asteps=_step)).shape[0]) 
             for cls_name, run in data_dict.items() 
             for run_name, dados in run.items()]
        )
        data = np.concatenate(
            [rolling_window(dados,_tam, asteps=_step) 
             for cls_name, run in data_dict.items() 
             for run_name, dados in run.items()], axis=0)
    elif tipo== "subtempo":
        label = np.concatenate(
            [trgt[cls_name]*np.ones(
                ((generate_subtempo(
                    rolling_window(dados,_tam, asteps=_step), window_size=window_size,
                                                      window_step=window_step,
                                                      trgt_multiplier=trgt[cls_name])
                 ).shape[0],1)
            )
             for cls_name, run in data_dict.items() 
             for run_name, dados in run.items()]
        )
        data = np.concatenate(
            [generate_subtempo(rolling_window(dados,_tam, asteps=_step), window_size=window_size,
                            window_step=window_step,
                            trgt_multiplier=trgt[cls_name]) 
             for cls_name, run in data_dict.items() 
             for run_name, dados in run.items()], axis=0)
    # if label_tipo == "24classes":
    #    
    #     label = np.concatenate(
    #         [trgt[cls_name]*np.ones((rolling_window(np.asarray(dados).reshape(-1),_tam, asteps=_step)).shape[0]) 
    #          for cls_name, run in data_dict.items() 
    #          for run_name, dados in run.items()])
    #        
    #     data = np.concatenate(
    #         [rolling_window(np.asarray(dados).reshape(-1),_tam, asteps=_step) 
    #          for cls_name, run in data_dict.items() 
    #          for run_name, dados in run.items()], axis=0)
    # else:

    return data, label

def separar_run(data_dict, _tam, _step, navios, trgt = None):
    if trgt is None:
      trgt = {
          'Class1': 0,
          'Class2': 1,
          'Class3': 2,
          'Class4': 3
          }
    y_train = np.concatenate(
        [trgt[cls_name]*np.ones((rolling_window(dados,_tam, asteps=_step)).shape[0]) 
        for cls_name, run in data_dict.items() 
        for run_name, dados in run.items() if run_name not in navios]
        )
    y_test = np.concatenate(
        [trgt[cls_name]*np.ones((rolling_window(dados,_tam, asteps=_step)).shape[0]) 
        for cls_name, run in data_dict.items() 
        for run_name, dados in run.items() if run_name in navios]
        )
    x_train = np.concatenate(
        [rolling_window(dados,_tam, asteps=_step)
         for cls_name, run in data_dict.items() 
         for run_name, dados in run.items() if run_name not in navios], axis=0
         )
    x_test = np.concatenate(
        [rolling_window(dados,_tam, asteps=_step)
         for cls_name, run in data_dict.items() 
         for run_name, dados in run.items() if run_name in navios], axis=0
         )
    return x_train, y_train, x_test, y_test 
