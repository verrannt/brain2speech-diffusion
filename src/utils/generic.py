from multiprocessing import Pool
import os
from sys import getsizeof
from pathlib import Path
import re

import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import detect_silence
import numpy as np
from tqdm import tqdm



def varinfo(var):
    """ Get info about type and size of a variable """
    bytes = getsizeof(var)
    print(f'Type: {type(var)}')
    print("Size: ", end="")
    if (bytes < 1e3):
        print(f"{bytes} Bytes")
    elif (bytes < 1e6):
        print(f"{round(bytes/1e3, 2)} kB ({bytes} Bytes)")
    elif (bytes < 1e9):
        print(f"{round(bytes/1e6, 2)} MB ({bytes} Bytes)")
    else:
        print(f"{round(bytes/1e9, 2)} GB ({bytes} Bytes)")


def sizeof_fmt(num, suffix='B'):
    """ by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def getmeminfo(
    items, # Need to pass locals().items() from outside this function's scope, else vars are not accessed
    keep_temp_vars: bool = False, # Whether to ignore temp vars that start with underscore
    show_all: bool = False,
    output_limit: int = 20
):
    # Collect names and sizes of variables
    collected_vars = sorted(
        (
            (name, getsizeof(value)) for name, value in items 
            if name[0] != '_' or keep_temp_vars #locals().items()
        ),
        key = lambda x: -x[1]
    )

    for name, size in collected_vars[:output_limit if not show_all else -1]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    if not show_all:
        print(f"{' '*20}(+ {len(collected_vars)-output_limit} more)")


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_silence(filename, min_silence_len=250, silence_thresh=-16):
    audio = AudioSegment.from_file(filename)#, format=str(filename).split('.')[-1])
    print(f'Read audio of length {int(audio.frame_count())} with sampling rate {audio.frame_rate}')

    silence_idx = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    print('Silent sections found at milliseconds:')
    print(silence_idx)

    audio_data = audio.get_array_of_samples()
    plt.plot(audio_data)
    plt.vlines([(section[0]/1000)*audio.frame_rate for section in silence_idx], ymin=-audio.max, ymax=audio.max, color='g')
    plt.vlines([(section[1]/1000)*audio.frame_rate for section in silence_idx], ymin=-audio.max, ymax=audio.max, color='r')
    plt.show()

    display(audio)


def calc_receptive_field_size(d:int, k:int, L:int):
    """
    Convenience method to calculate the receptive field size of the DiffWave
    model. This is needed to check if the segment length used in training
    is actually covered fully by the receptive field.

    d : length of one dilation cycle
    k : kernel size
    L : number of residual layers
    """
    return (k-1) * np.sum([2**(l%d) for l in range(1, L+1)]) + 1


def get_random_indexes(N:int, train_split:float=0.8, val_split:float=0.0, subset:float=1.0, seed:int=None):
    """
    Get random indexes for train, test, and validation split. Test split is
    computed as 1.0 - (train_split + val_split). The `subset` parameter allows
    to only use a given percentage of random samples.
    """
    assert 0.0 < train_split + val_split <= 1.0
    assert 0.0 < subset <= 1.0

    rng = np.random.default_rng(seed)

    # Draw random indexes before potentially subsetting
    rand_idx = rng.choice(N, size=N, replace=False)

    # Compute amount of indexes to actually return, based on given
    # subset percentage
    N = int(N * subset)

    test_split = 1.0 - (train_split + val_split)

    train_size = int(train_split * N)
    val_size = int(val_split * N)
    test_size = N - val_split - train_split

    train_idx = rand_idx[:train_size]
    val_idx = rand_idx[train_size:train_size+val_size]
    test_idx = rand_idx[train_size+val_size:N]

    return train_idx, val_idx, test_idx


def get_files_in_dir(path: str, filetype: str):
    """ 
    List all files of type `filetype` in directory given by `path`. 
    
    Example:
    ```
    files = get_files_in_dir('path/to/data', 'wav')
    ```
    """
    all_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(f'.{filetype}'):
                all_files.append(os.path.join(root, file))
    return all_files


def get_word_from_filepath(filepath: str, uses_augmentation: bool = True, uses_numbering: bool = True) -> str:
    """
    Extract the word from a given filepath pointing to an audio file on disk.
    
    Example:
    ```
    get_word_from_filepath('path/to/file/brief_pitch1.wav')
    > 'brief'
    ```
    """
    # Get the last part of the path (i.e. just the filename)
    filepath = filepath.split('/')[-1]
    # Get the filename before the file extension
    filepath = filepath.split('.')[0]
    # Augmented files are named according to {word}_{aug-type}.wav, so this removes the augmentation from the name
    if uses_augmentation:
        filepath = filepath.split('_')[0]
    # Some files (e.g. ECoG files) are numbered (e.g. goed1.npy), so this removes any digits from the name
    if uses_numbering:
        filepath = re.sub(r'[0-9]', '', filepath)
    return filepath


def multi_proc(
    iterator,
    F,
    total: int,
    num_processes: int = 1,
    postprocess: str = None,
    verbose: bool = True,
):
    """
    Execute function `F` on each element in `iterator` using a pool of 
    `num_processes` workers, and return the result.

    Since it is possible for `F` to return a single or multiple values,
    setting the `postprocess` argument allows for quick restructuring of
    the generated result list.

    If the result list is nested (2-D), pass `postprocess='flatten'` to
    flatten it into a 1-D list.
    (Example: [[1,2,3], [1,2,3]] --> [1,2,3,1,2,3])

    If the individual items are to be unpacked into separate lists, pass
    `postprocess='unpack'` to unpack and return several distinct lists.
    (Example: [[1,2,3], [1,2,3]] --> [1,1],[2,2],[3,3])

    In case the chosen postprocessing method does not work (e.g. when `F`
    returns only a single element and the result list is thus 1-D, trying to 
    flatten would cause an exception), all exceptions are caught and the
    unaltered result list is returned.

    Params
    ---
    iterator : iterable on each element of which the function `F` should be
        executed
    F : function that takes as input and element of `iterator` and returns
    total : total number of items that are in the `iterator`
    num_processes : number of processes to spawn for multiprocessing
    postprocess : `flatten`, `unpack`, or `None`. Which postprocessing of 
        the result list to apply
    verbose : Wether to show progress bar or not

    Returns
    ---
    A list with results from applying `F` to the elements of `iterator` in 
    the same order of elements in `iterator`.
    """

    with Pool(num_processes) as p:
        res_list = [
            result for result in tqdm(p.imap(F, iterator), total=total, disable=not verbose)
        ]

    try:
        if postprocess is None:
            return res_list
        elif postprocess == 'flatten':
            return [item for sublist in res_list for item in sublist]
        elif postprocess == 'unpack':
            return list(zip(*res_list))
        else:
            print(f"Unrecognized postprocess argument {postprocess}. " 
                    "Returning result list unchanged.")
            return res_list
    except Exception as e:
        # If anything goes wrong, e.g. because a 1-D list has been tried
        # to be flattened, simply return the default results
        print("There was an exception postprocessing:")
        print(e)
        print("Will return the default unaltered results")
        return res_list

def fix_length(array, desired_length: int):
    """ 
    Fix the length of `array` to `desired_length` in first dimension. 
    If `array` is longer, it will be clipped, if it is shorter, it will be 
    padded with zeros.
    """
    if array.shape[0] > desired_length:
        return array[:desired_length]
    elif array.shape[0] < desired_length:
        return np.concatenate([
            array, 
            np.zeros(desired_length - array.shape[0])
        ], axis=0)
    else:
        return array
