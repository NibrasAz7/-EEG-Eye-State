# -EEG-Eye-State

Goal: Determain the eye state (Open/Close) based on EEG signal.

DatasetSource:
Oliver Roesler, it12148 '@' lehre.dhbw-stuttgart.de , Baden-Wuerttemberg Cooperative State University (DHBW), Stuttgart, Germany
https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State

Features of Emotiv EEG Neuroheadset:
14 channels
Rigid Electrode Placement (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
Wet Electrodes (require saline water)
2048 Hz Sampling Rate
14 bits

Dataset Details:
All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. The duration of the measurement was 117 seconds. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analysing the video frames. '1' indicates the eye-closed and '0' the eye-open state. All values are in chronological order with the first measured value at the top of the data.

Note: Measurment duration is 117 seconds with 14980 sample point. Therfore, Sampling Frequency equels to: 14980/117 = 128 Hz.
The data was downsampled from 2048 Hz to 128



