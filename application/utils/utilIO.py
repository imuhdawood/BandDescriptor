import numpy as np
import inspect
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm
from skimage.io import imread, imsave
from termcolor import colored
import pickle
import pandas as pd
import argparse
import pathlib

def rm_n_mkdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return


def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return


def recur_find_ext(root_dir, ext_list):
    """
    recursively find all files in directories end with the `ext`
    such as `ext='.png'`

    return list is alrd sorted
    """
    assert isinstance(ext_list, list)
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)

def writePickle(ofile,G):
    with open(ofile, 'wb') as f:
          pickle.dump(G, f)
