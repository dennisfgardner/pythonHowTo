"""this example uses ECG data obtained from Kaggle:
https://www.kaggle.com/datasets/devavratatripathy/ecg-dataset
"""

import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class ecgDataHandler():

    def __init__(self):
        self.ecg_data = np.genfromtxt('./data/ecg.csv', delimiter=',')

    def get_patient_ecg(self, patient_number):
        """traces need to be flipped, hence minus sign"""
        return -1*self.ecg_data[patient_number, :-1]

    def is_ecg_normal(self, patient_number):
        return True if self.ecg_data[patient_number, -1] == 1.0 else False


def main():
    # pick a ecg patient number (0-4997)
    patient_number = random.randint(0, 4997)
    patient_number = 2466

    # load the data
    ecg_data_handler = ecgDataHandler()

    # get ecg
    ecg = ecg_data_handler.get_patient_ecg(patient_number)

    # convert True/False into Normal/Abnormal
    ecg_status = 'Normal' if ecg_data_handler.is_ecg_normal(patient_number) \
        else 'Abnormal'

    # find the peak by selecting largest peak
    peaks, _ = find_peaks(ecg, prominence=1)
    peak = peaks[np.argmax(ecg[peaks])]

    plt.style.use('seaborn-poster')
    fig, ax = plt.subplots(1, 1)
    ax.plot(ecg)
    ax.plot(peak, ecg[peak], 'X')
    ax.set_title(f'ECG for Patient {patient_number} is {ecg_status}')
    plt.show()


if __name__ == '__main__':
    main()
