"""
.. module:: MFT

MFT
*************

:Description: MFT

    Computes the Momentary Fourier Transformation

    Computes the n firsts coefficients of the FFT for consecutive windows given the n first FFT coeffs of the first window
    (cost is O(nwindows*ncoef) instead of  O(wsize*log(wsize)*nwindows)

    Albrecht, Cumming, Dudas "The Momentary Fourier Transformation Derived from Recursive Matrix Transformation"

@inproceedings{albrecht1997momentary,
  title={The momentary fourier transformation derived from recursive matrix transformations},
  author={Albrecht, S and Cumming, I and Dudas, J},
  booktitle={Digital Signal Processing Proceedings, 1997. DSP 97., 1997 13th International Conference on},
  volume={1},
  pages={337--340},
  year={1997},
  organization={IEEE}
}

:Authors: bejar
    

:Version: 

:Created on: 15/02/2017 13:48 

"""
import numpy as np
import time

__author__ = 'bejar'


def mft(series, sampling, ncoef, wsize, butfirst=False):
    """
    Computes the ncoef fourier coefficient for the series

    butfirst eliminates the first coefficient (mean of the signal)

    :return:
    """

    if butfirst:
        ncoef += 1
    nwindows = len(series) - wsize
    # imaginary matrix for the coefficients
    coef = np.zeros((nwindows, ncoef), dtype=np.complex)
    dcoef = np.zeros(ncoef, dtype=np.complex)
    fwsize = float(wsize)
    pi2i = 1j * 2 * np.pi

    y = np.fft.rfft(series[:wsize])
    for l in range(ncoef):
        coef[0, l] = y[l]
        dcoef[l] = np.exp(pi2i * (l / fwsize))

    for w in range(1, nwindows):
        coef[w] = dcoef * (coef[w - 1] + (series[wsize + (w - 1)] - series[w - 1]))

    if butfirst:
        return coef[:, 1:]
    else:
        return coef



if __name__ == '__main__':
    pass
    # itime = time.time()
    # coef1 = mft(signal, sampling=10, ncoef=15, wsize=32)
    # ftime = time.time()
    # print(ftime - itime)

