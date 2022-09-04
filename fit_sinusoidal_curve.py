""" Fit Sinusoidal Curve

This code is adapted from unsym's answer to a stack overflow question, "How do
I fit a sine curve to my data with pylab and numpy?".
https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy

"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def sinfunc(x, amp, omega, phi, offset):
    y = amp * np.sin(omega*x + phi) + offset
    return y


def fit_sin(x, y):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc
    "'''

    # assume uniform spacing
    fx = np.fft.fftfreq(len(x), (x[1]-x[0]))
    # Fourier transform of y
    Fy = abs(np.fft.fft(y))
    # get max frequency that's not the DC peak
    guess_freq = abs(fx[np.argmax(Fy[1:])+1])
    guess_omega = 2.*np.pi*guess_freq
    # use the relationship between sine's RMS and amplitude, we only want the
    # amplitude from zero, because we have an offset term, so subtract the
    # mean first. subtracting the means results in std() and RMS are the same
    guess_amp = 2.0**0.5*np.std(y)
    guess_offset = np.mean(y)
    # TODO find a way to estimate phi, the phase offset
    guess_phi = 0

    print(f'{guess_amp=:.2f}\t{guess_omega=:.2f}\t{guess_phi=:.2f}\t'
          f'{guess_offset=:.2f}')

    # optimal parameters (popt) that minimize the sum of the squared
    # residuals and the estimated covariance of popt
    popt, _ = scipy.optimize.curve_fit(
        sinfunc, x, y,
        p0=[guess_amp, guess_omega, guess_phi, guess_offset]
    )
    fit_amp, fit_omega, fit_phi, fit_offset = popt
    return fit_amp, fit_omega, fit_phi, fit_offset


def main():
    # number of samples
    num_pts = 512
    # amplitude
    amp = 1
    # angular frequency (2*pi*f) in units of radians per second
    omega = 2
    # phase offset in radians
    phi = 0.5
    # DC offset
    offset = 4
    # noise
    noise = 1

    print(f'{amp=:.2f}\t{omega=:.2f}\t{phi=:.2f}\t{offset=:.2f}')

    # create sine wave with noise
    x = np.linspace(0, 10, num_pts)
    y = sinfunc(x, amp, omega, phi, offset)
    ynoise = y + noise*(np.random.random(len(x))-0.5)

    # fit noisy sinusoidal curve
    fit_amp, fit_omega, fit_phi, fit_offset = fit_sin(x, ynoise)
    yfit = sinfunc(x, fit_amp, fit_omega, fit_phi, fit_offset)

    print(f'{fit_amp=:.2f}\t{fit_omega=:.2f}\t{fit_phi=:.2f}\t'
          f'{fit_offset=:.2f}')

    plt.style.use('seaborn-poster')
    _, ax = plt.subplots()
    ax.plot(x, y, label='perfect y')
    ax.plot(x, ynoise, label='noisy y')
    ax.plot(x, yfit, label='fit y')
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_title("Simple Plot")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
