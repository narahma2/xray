# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:05:35 2018

@author: rahmann
"""
import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

start = time.time()
project = "D:/Naveed/2018 APS Imaging"

dark = np.array(Image.open(project + "/Processed/Dark1001.tif"))

flat_path = project + "/Raw/Flat/Flat1/"
flat_list = [flat_path + f for f in os.listdir(flat_path) if f.endswith('.tif')]
flat = []

for i in range(len(flat_list)):
    flat.append(np.float32(np.array(Image.open(flat_list[i]))))
    
flat_path = project + "/Raw/Flat/FlatSi/"
flat_list = [flat_path + f for f in os.listdir(flat_path) if f.endswith('.tif')]
flatSi = []

for i in range(len(flat_list)):
    flatSi.append(np.float32(np.array(Image.open(flat_list[i]))))
    
del flat_path, flat_list, i

jet_path = project + "/Raw/Jet/"
jet_folders = [jet_path + f for f in os.listdir(jet_path)]

for i in range(len(jet_folders)):
#for i in range(1):
    jet_list = [jet_folders[i] + "/" + f for f in os.listdir(jet_folders[i]) if f.endswith('.tif')]
    jet = []
    
    for j in range(len(jet_list)):
        jet.append(np.array(Image.open(jet_list[j])))
    
    save_folder = project + "/Processed/h5chunk"
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    h5f = h5py.File(str.replace(jet_folders[i], "/Raw/Jet", "/Processed/h5chunk") + ".h5", 'w')

    for j in range(len(jet)):
        if "Fn" in jet_folders[i]:
            denom = flat[j] - dark
            if np.count_nonzero(denom == 0):
                denom[denom == 0] = np.nan
            normalized = (jet[j] - dark) / denom
            h5f.create_dataset(os.path.splitext(os.path.basename(jet_list[j]))[0], data=normalized, dtype=np.float32, chunks=True, compression='lzf')
        else:
            denom = flatSi[j] - dark
            if np.count_nonzero(denom == 0):
                denom[denom == 0] = np.nan
            normalized = (jet[j] - dark) / denom
            h5f.create_dataset(os.path.splitext(os.path.basename(jet_list[j]))[0], data=normalized, dtype=np.float32, chunks=True, compression='lzf')
    h5f.close()

end = time.time()
print(end - start)
