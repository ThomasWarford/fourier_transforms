#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script plots functions and their respective fourier transforms.
Created on Sun Oct 31 14:26:30 2021

@author: Thomas Warford
"""

import numpy as np
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

def top_hat(time):
    '''
    Top hat function.

    Parameters
    ----------
    time : float

    Returns
    -------
    float

    '''
    return np.where((time > -0.5) & (time < 0.5), 1, 0)

def guassian(time):
    '''
    Guassian function.

    Parameters
    ----------
    time : float

    Returns
    -------
    float

    '''
    return np.exp(-time ** 2)

def sinc(time):
    '''
    Sinc function.

    Parameters
    ----------
    time : float

    Returns
    -------
    float

    '''
    return np.sin(time) / time

def damped_cosine(time):
    '''
    Exponential damped cosine function.

    Parameters
    ----------
    time : float

    Returns
    -------
    float

    '''
    return np.where(time > 0, np.exp(-time) * np.cos(time), 0)

def cosine(time):
    '''
    Cosine function.

    Parameters
    ----------
    time : float

    Returns
    -------
    float

    '''
    return np.cos(time)

functions = [top_hat, guassian, sinc, damped_cosine, cosine]

NUMBER_OF_STEPS = 100000
TIME_BOUND = 10
time_step = TIME_BOUND / NUMBER_OF_STEPS

time_ = np.linspace(-TIME_BOUND, TIME_BOUND, NUMBER_OF_STEPS)

figure, axes = plt.subplots(2, len(functions), figsize=(24, 13.5))

frequencies = fftfreq(NUMBER_OF_STEPS, time_step)

for counter, function in enumerate(functions):
    amplitude = function(time_)
    axes[0, counter].plot(time_, amplitude)
    axes[0, counter].set_title(f'f(t) = {function.__name__}(t)')

    fourier_transform = fft(fftshift(amplitude))
    axes[1, counter].plot(frequencies, fourier_transform)
    axes[1, counter].set_xlim([-5, 5])
    axes[1, counter].set_title('F(frequency)')
