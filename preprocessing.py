#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:12:53 2022



One component of the Spectral Detect project.


Spectral Detect was developed in under 2 months as part of the Professional Master of Computer Science 
degree which I worked on in a team of three. We worked with the Department of Conservation
to prototype an application to detect Euphorbia paralias (sea spurge, highly invasive beach plant)
in hyperspectral images. 


This module highlights solving a difficult problem. The hyperspectral images were up to 500 Gb and we 
had limited compute resources, so out of memory errors were common.
I solved it by using the TensorFlow Dataset object, which allows the use of a generator to generate the
tensor elements from the dataset (which is loaded as a memmap). 
Process.data_generator yields chunks of data after applying Factor Analysis, and when used to create 
a Dataset object we can map a function on the whole 500 Gb dataset! 
So we map the method Process.unfold_data which creates a window for each pixel, and then we can 
infer on each chunk and save the results. 

(the code could use a refactor, but we were under time pressure with many features to ship)



The model, although not included, was adapted from SpectralNET. 
It is very interesting work, you should check it out.
https://github.com/tanmay-ty/SpectralNET
"""
import os
from os.path import exists
import pathlib
import numpy as np
import pickle as pk
import spectral
import spectral.io.envi as envi
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.image import extract_patches
import tensorflow as tf
# from predict import Predict




def applyFA(data, num_components=75, max_iter=4000):
    """ Apply Function Analysis from scikit learn to hyper spectral data cube

    Parameters
    ----------
    data : array_like
        hyperspectral array or memmap to be decomposed
    num_components : int, optional
        Number of dimensions to reduce to, by default 75
    max_iter : int, optional
        Max number of iterations before stopping

    Returns
    -------
    array_like, FactorAnalysis
        the factor reduced array_like result of the factor analysis
        the fitted FactorAnalysis class object for saving
    """
    decomposed_data = np.reshape(data, (-1, data.shape[2]))
    fa = FactorAnalysis(n_components=num_components, random_state=0, max_iter=max_iter)
    decomposed_data = fa.fit_transform(decomposed_data)
    decomposed_data = np.reshape(decomposed_data, (data.shape[0],data.shape[1], num_components))
    return decomposed_data, fa


def applyPCA(data, num_components=75):
    """Apply Principal Component Analysis from scikit learn to hyper spectral data cube

    Parameters
    ----------
    data : array_like
        hyperspectral array or memmap to be decomposed
    num_components : int, optional
        Number of dimensions to reduce to, by default 75

    Returns
    -------
    array_like, PCA
        array_like result of the top num_components from principal component analysis
        the fitted PCA class object for saving
    """
    decomposed_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    decomposed_data = pca.fit_transform(decomposed_data)
    decomposed_data = np.reshape(decomposed_data, (data.shape[0],data.shape[1], num_components))
    return decomposed_data, pca

def create_decomposition(data, num_components=75, method="fa"):
    """Create and fit decomposition of hyperspectral datacube then save fit to pickle file

    Parameters
    ----------
    data : array_like
        hyperspectral array or memmap to be decomposed
    num_components : int, optional
        Number of dimensions to reduce to, by default 75
    method : str, optional
        method is a string representation for which dimensionality reduction method to apply
        Can be "fa" for Functional Analysis or "pca" for Principal Component Analysis, by default "fa"

    Returns
    -------
    array_like
        array_like result of the decomposition
    """
    if method.lower() not in ["fa", "pca"]:
        raise ValueError("method variable can be 'fa' for Functional Analysis or 'pca' for Principal Component Analysis")
    if method.lower() == "fa":
        decomposed_data, decomposition = applyFA(data, num_components)
    elif method.lower() == "pca":
        decomposed_data, decomposition = applyPCA(data, num_components)

    pk.dump(decomposition, open(f"{method}-{num_components}.pkl", "wb"))
    return decomposed_data



def apply_decomposition(data, num_components=3, method="fa"):
    """Load decomposition fit from pickle file and apply it to the hyperspectral data.
    If pickle file doesn't exist, assume it needs to be created by call create_composition instead.
    Incoming data must have same number of channels as the data used to create the pickle

    Parameters
    ----------
    data : array_like
        hyperspectral array or memmap to be decomposed
    num_components : int, optional
        Number of dimensions to reduce to, by default 75
    method : str, optional
        method is a string representation for which dimensionality reduction method to apply
        Can be "fa" for Functional Analysis or "pca" for Principal Component Analysis, by default "fa"

    Returns
    -------
    array_like
        array_like result of the decomposition
    """
    if not exists(method + '.pkl'):
        print(f"{method}.pkl decomposition file does not exist, fitting new {method} instead")
        return create_decomposition(data, num_components, method)
    decomposition = pk.load(open(method + ".pkl", 'rb'))
    decomposed_data = tf.reshape(data, (-1, data.shape[2]))
    print(1, decomposed_data.shape)

    decomposed_data = decomposition.transform(decomposed_data)
    print(2, decomposed_data.shape)

    decomposed_data = tf.reshape(decomposed_data, (data.shape[0], data.shape[1], num_components))
    print(3, decomposed_data.shape)
    return decomposed_data



def pad_with_zeros(data, margin=2):
    """pad 3 dimensional array_like with zeros

    Parameters
    ----------
    data : array_like
        numpy array or memmap to be padded
    margin : int, optional
        amount of padding to apply, by default 2

    Returns
    -------
    array_like
        padded array
    """
    padded = np.zeros((data.shape[0] + 2 * margin, data.shape[1] + 2* margin, data.shape[2]))
    x_offset = margin
    y_offset = margin
    padded[x_offset:data.shape[0] + x_offset, y_offset:data.shape[1] + y_offset, :] = data
    return padded



class Preprocess:
    def __init__(self, job, window_size=24,job_type="envi"):
        """Performs the preprocessing jobs of opening the dataset,
        applying decomposition and breaking into smaller pieces if necessary

        Parameters
        ----------
        job : Job
            Incoming to Job to preprocess
        job_type : str, optional
            job_type in ["envi", "array", "mat"], by default "envi"
            type of spectral data structure. envi requires additional processing.
            mat (.mat matlab spectral file) out of scope
        window_size : int, optional
            Size of the window around the pixel to be predicted on
        job_type : str, optional
            "envi" if loading envi file or can be "array" for npy file
        """
        self.job = job
        self.window_size = window_size
        self.padding = 12
        self.job_type = job_type
        self.file_path = "data/"
        self.file_size = 200
        # self.model = Predict("testing_data/test_model.hdf5")
        # self.begin_process() #probably call this separately


    def begin_process(self):
        if self.job_type == "envi":
            HSI_data = self.load_envi()
        elif self.job_type == "array":
            HSI_data == self.load_array()
        return HSI_data

    def load_envi(self):
        """Load spectral data from envi header and binary file

        Returns
        -------
        array_like
            numpy memmap of spectral data
        """
        hdr_path = os.path.join(self.file_path, self.job.file)
        bin_path = os.path.join(self.file_path, self.job.file[:-4])
        file_ = envi.open(hdr_path, bin_path)
        HSI_data = file_.open_memmap(interleave="bip")
        print(f"Opened {self.job.filename} with shape {HSI_data.shape}")
        return HSI_data


    def load_array(self, file_name=None):
        """Load numpy .npy array file

        Parameters
        ----------
        file_name : str, optional
            filename to open, by default None
            Use self.job.file if no file_name specified.

        Returns
        -------
        array_like
            opened data array
        """
        if not file_name:
            file_name = self.job.file
        array_path = os.path.join(self.file_path, file_name)
        HSI_data = np.load(array_path)
        return HSI_data



class Process:
    """
    Load the hyperspectral image into a memmap and create tf.data.Dataset object.
    Perform inference using SpectralNet and save the results
    """
    def __init__(self, data, stride=24, window_size=240, num_components=3):
        self.data = data
        self.stride = stride
        self.window_size = window_size
        self.height, self.width, self.depth = data.shape
        self.num_components = num_components
        self.data_type = tf.float32
        self.in_shape = data.shape
        self.out_shape = (1, window_size, window_size, num_components)
        self.model = None # must attach model to run

    def attach_model(self, model):
        """
        Link the tensorflow model, makes Process easier to initialize
        """
        self.model = model

    def data_generator(self):
        """
        Yields chunks of size window_size by window_size after applying FA/PCA
        """
        y, x = self.height, self.width
        w = self.window_size
        for i in range(y//w):
            for j in range(x//w):
                yield np.expand_dims(apply_decomposition(self.data[i*w:i*w+w, j*w:j*w+w,]), 0)

    def unfold_data(self, data):
        """
        Creates a sliding window of size stride by stride for each pixel
        This enables pixel by pixel detection using both spectral and spatial
        properties
        """
        patches = extract_patches(images=data,
                            sizes = [1, self.stride, self.stride, 1],
                            strides=[1, 1, 1, 1],
                            rates=[1, 1, 1, 1],
                            padding="SAME")
        return tf.reshape(
                    patches,
                        (self.window_size*self.window_size,
                        self.stride,
                        self.stride,
                        self.num_components))


    def run_predict(self, ary, filepath=None):
        """
        Perform inference and save to filepath
        """
        # ary = np.load("testing_data/ary.npy")
        print(f"test data shape: {ary.shape}")
        if not filepath:
            filepath = "data/results/predictions"
        self.model.predict_classes(filepath, ary)

    def process_data(self, user, filename):
        """
        Create dataset from data_generator, map unfold_data onto it, 
        and infer with run_predict
        """
        dataset = tf.data.Dataset.from_generator(
            generator=self.data_generator,
            output_types=self.data_type,
            output_shapes=self.out_shape
        )
        print("Creating dataset from generator")

        dataset = dataset.map(self.unfold_data)

        folder = 'data/' + user + '/predictions/'
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for i, ex in enumerate(dataset):
            print(i, ex.shape, ex.dtype)
            self.run_predict(ex, folder + filename[:-4] + "_" + str(i))


def restore_folder_to_array(user, filename, HSI_shape, window_size):
    """
    Combine all images in results folder into one array
    """
    results_folder = 'data/' + user + '/results/'
    preds_folder = 'data/' + user + '/predictions/'
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    shape = (HSI_shape[0], HSI_shape[1])
    new_data = np.zeros(shape)
    height, width = shape
    w0 = window_size
    x = 0
    for i in range(height//window_size):
        for j in range(width//window_size):
            ary = np.load(f"{preds_folder}{filename}_{x}.npy").reshape((window_size, window_size))
            new_data[i*w0:i*w0+w0, j*w0:j*w0+w0] = ary
            x += 1
    np.save(results_folder + 'allpoints', new_data)
    return new_data
