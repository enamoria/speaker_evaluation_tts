# -*- coding: utf-8 -*-

""" Created on 9:27 AM, 4/10/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь!
"""

import logging
import os
import pickle
import subprocess
import scipy
import parmap

import librosa as lbr
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw

from scipy.stats import describe
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Pool, cpu_count

from pyEval.utils import read_dictionary, get_data

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

cpu_rate = 0.8
# frame_length = 25  # ms
# hop_length = 10  # ms

is_toy = False

speaker = "doanngocle_1" if not is_toy else "doanngocle_1_toy"


def _calculate_mean_sd_energy(raw_wav, n_fft, n_hop):
    """
    Calculate 1
    :param _:
    :return:
    """
    energy = []

    for count, i in enumerate(range(0, len(raw_wav) - n_fft, n_hop)):
        # temp = np.multiply(raw_wav_windowed, raw_wav[i: i + n_fft])
        energy.append(np.sqrt(np.sum(np.square(raw_wav[i: i + n_fft]))))

    _energy = np.sum(energy) / len(energy)
    temp = scipy.stats.describe(energy)

    return temp.mean, np.sqrt(temp.variance)


def calculate_mean_sd_energy(raw_wavs, sr, frame_length, hop_length):
    """
    Calculate mean & sd for whole dataset
    :param _:
    :return:
    """
    logging.info("Calculating energy data ...")

    n_fft = int(frame_length * sr / 1000)
    n_hop = int(hop_length * sr / 1000)

    energies = []
    for wav in tqdm(raw_wavs):
        energies.append(_calculate_mean_sd_energy(wav, n_fft, n_hop))

    return energies


def _calculate_f0(raw_wav, sr, frame_period):
    f0, _ = pw.harvest(raw_wav, sr, frame_period)
    return f0


def calculate_f0(raw_wavs, sr, frame_period=5, parallel=True):
    f0s = []
    logging.info("Calculating f0 ...")

    if not parallel:
        for wav in raw_wavs:
            f0s.append(_calculate_f0(wav, sr, frame_period))
    else:
        n_worker = int(cpu_count() * cpu_rate)
        logging.info("{}% resources ({} cpu core(s)) will be used for f0 calculation".format(cpu_rate * 100, n_worker))
        # f0s = pool.starmap(_calculate_f0, zip(raw_wavs, repeat(sr), repeat(frame_period)))
        f0s = parmap.starmap(_calculate_f0, list(zip(raw_wavs, repeat(sr), repeat(frame_period))), pm_processes=n_worker, pm_pbar=True)

    return f0s


def calculate_mean_sd_f0(f0s):
    sum_f0s = []

    for i in range(len(f0s)):
        not_silence_f0 = [value for value in f0s[i] if value != 0]
        sum_f0 = sum(not_silence_f0) / len(not_silence_f0)

        sum_f0s.append(sum_f0)

    return sum_f0s


def _calculate_speaking_rate(raw_wav, text, sr, dictionary):
    """
        Speaking rate equals to number of `phonemes / length (sec) in a sentence`
        `dictionary` will reused pyGenLab resources :D

        Text should be normalized at first
    :param text:
    :param length_wav:
    :param dictionary:
    :return:
    """
    phoneme_count = 0
    for word in text.split(" "):
        try:
            phoneme_count += len([xxx for xxx in dictionary[word] if not xxx.isdigit()])
        except KeyError as ke:
            logging.error("{}: {} can't be found in dictionary".format(ke, word))

    length_wav = len(raw_wav) / sr
    return phoneme_count / length_wav


def calculate_speaking_rate(raw_wavs, texts, sr, dictionary):
    n_worker = int(cpu_count() * cpu_rate)
    logging.info("{}% resources ({} cpu core(s)) will be used for speaking_rate calculation".format(cpu_rate * 100, n_worker))

    # rates = pool.starmap(_calculate_speaking_rate, zip(raw_wavs, texts, repeat(sr), repeat(dictionary)))
    rates = parmap.starmap(_calculate_speaking_rate, list(zip(raw_wavs, texts, repeat(sr), repeat(dictionary))), pm_processes=n_worker, pm_pbar=True)

    return rates


def _calculate_mean_utt_length(raw_wav, sr):
    """
        Calculate utterance length (in second)
    :param raw_wav:
    :param sr:
    :return:
    """
    return len(raw_wav) / sr


def calculate_mean_utt_length(raw_wavs, sr):
    n_worker = int(cpu_count() * cpu_rate)
    logging.info("{}% resources ({} cpu core(s)) will be used for mean utterance length calculation".format(cpu_rate * 100, n_worker))

    # return pool.starmap(_calculate_mean_utt_length, zip(raw_wavs, repeat(sr)))
    return parmap.starmap(_calculate_mean_utt_length, list(zip(raw_wavs, repeat(sr))), pm_processes=n_worker, pm_pbar=True)


def draw_plot(energy_data=None, f0_data=None, speaking_rate_data=None, utt_length=None):
    """
        Draw plot of whichever-provided data
    :param energy_data:
    :param f0_data:
    :param speaking_rate_data:
    :param utt_length:
    :return:
    """
    logger_plot = logging.getLogger("eval")
    # energy

    if not energy_data and not f0_data and not speaking_rate_data and not utt_length:
        logger_plot.critical("None of kwargs is specified. Nothing to plot")
        return -1
    else:
        logger_plot.info("Presenting plot ...")

        if energy_data:
            yyy = [xx[0] for xx in energy_data]  # mean
            xxx = [xx[1] for xx in energy_data]  # var

            fig, axes = plt.subplots(1, 3, sharey=True)
            fig.suptitle("Energy data")

            axes[0].scatter(xxx, yyy)
            axes[0].set_xlabel("Standard deviation")
            axes[0].set_ylabel("Mean")

            axes[1].boxplot(xxx)
            axes[1].set_ylabel("Mean")

            axes[2].boxplot(yyy)
            axes[2].set_ylabel("SD")

        if f0_data:
            fig, axes = plt.subplots(1, 2)
            fig.suptitle("f0 data")

            axes[0].hist(f0_data, 50)
            axes[0].set_xlabel("f0")

            axes[1].boxplot(f0_data)
            axes[1].set_ylabel("f0")

        if speaking_rate_data:
            fig, axes = plt.subplots(1, 2)
            fig.suptitle("speaking_rate: how fast she/he is doing")

            axes[0].hist(speaking_rate_data, 50)
            axes[0].set_xlabel("speaking rate (phonemes/sec)")

            axes[1].boxplot(speaking_rate_data)
            axes[1].set_ylabel("speaking_rate")

        if utt_length:
            fig, axes = plt.subplots(1, 2)
            fig.suptitle("length of utterance (sec)")

            axes[0].hist(utt_length, 50)
            axes[0].set_xlabel("utt length (sec)")

            axes[1].boxplot(utt_length)
            axes[1].set_ylabel("utt length")

    plt.show()


def _calculate(data, sr, speaker, output_path, dictionary_path, frame_length, hop_length):
    """
    Calculate some measurements for speaker evaluation (and his/her synthesized voice)
    These below attributes will be taken into account:
        - Mean & SD of `energy` for the whole dataset
        - Mean & SD of `f0` for `voiced frames` for the whole dataset
        - Mean Speaking rate ( each = number of phonemes / length (sec) in a sentence) for the whole dataset
        - Mean utterance length (sec) for the whole dataset
    Each attribute will be implemented separately. This function packages all of them into a single call

    :param data: a list, which is read from `get_data`: a list of [filename, text, wav]
    :param sr: sampling rate
    :param speaker: speaker name
    :param dictionary_path: path of the phonetic dictionary, used for speaking_rate calculation
    :param output_path: where to save the `features` object
    :param frame_length: length of a frame (sec)
    :param hop_length: stride (sec)
    :return:
    """
    feature_output = os.path.join(output_path, speaker) + ".pkl"

    dictionary = read_dictionary(dictionary_path)

    filenames = []
    texts = []
    wavs = []

    for i in range(len(data)):
        filenames.append(data[i][0])
        texts.append(data[i][1])
        wavs.append(data[i][2])

    # Energy
    energies = calculate_mean_sd_energy(wavs, sr, frame_length, hop_length)

    # f0
    f0s = calculate_f0(wavs, sr, frame_period=10)
    sum_f0s = calculate_mean_sd_f0(f0s)

    # Speaking rate
    speaking_rates = calculate_speaking_rate(wavs, texts, sr, dictionary)

    # Utterance length
    utt_length = calculate_mean_utt_length(wavs, sr)

    # Feature dictionary
    features = {
        'energy': energies,
        'f0': sum_f0s,
        'speaking_rates': speaking_rates,
        'utt_length': utt_length
    }

    subprocess.call(['mkdir', '-p', output_path])

    with open(feature_output, "wb") as f:
        pickle.dump(features, f)

    return features


def calculate(speaker, input_path, output_path="features", dictionary_path="vn.dict", frame_length=25, hop_length=10, recalculate=False):
    """
        Read data and pass to _calculate
    :param input_path:
    :param output_path:
    :param dictionary_path:
    :param frame_length:
    :param hop_length:
    :param recalculate:
    :return:
    """
    feature_output = os.path.join(output_path, speaker) + ".pkl"

    if os.path.exists(feature_output) and not recalculate:
        logging.info("Found a pre-calculation of feature for speaker `{}` in `{}`.".format(speaker, feature_output))
        logging.info("Either delete the file or pass `recaculate=True` to this function to re-calculate feature")

        with open(feature_output, "rb") as f:
            features = pickle.load(f)
            return features
    else:
        data, sr = get_data(os.path.join(input_path, speaker))
        features = _calculate(data, sr, speaker, output_path, os.path.join(os.path.dirname(os.path.abspath(__file__)), dictionary_path), frame_length, hop_length, recalculate)

    return features


if __name__ == "__main__":
    # data, sr = get_data(os.path.join("/data/data/tts/", speaker))

    # features = _calculate(data, sr, speaker)
    features = calculate(speaker, input_path="/data/data/tts/")

    energies, sum_f0s, speaking_rates, utt_length = features['energy'], features['f0'], features['speaking_rates'], features['utt_length']

    # Plot
    draw_plot(energy_data=energies, f0_data=sum_f0s, speaking_rate_data=speaking_rates, utt_length=utt_length)

    print("Done")
