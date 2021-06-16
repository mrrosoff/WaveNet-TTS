
import argparse
from yaml import safe_load as loadYAML

from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf

from librosa import load as loadAudio
from librosa.feature import rms
from librosa.core import frames_to_samples

from os import listdir, mkdir
from os.path import join, exists, splitext
from re import search
from random import shuffle
from threading import Thread

from WaveNet import calculate_receptive_field


def mu_law_encode(audio, quantization_channels=256):

    mu = tf.constant(quantization_channels - 1, dtype=tf.float32)
    safe_audio_abs = tf.math.minimum(tf.math.abs(audio), 1.0)  # min(|x|, 1)
    # log_e(1 + mu * min(|x|, 1)) / log_e(1 + mu)
    magnitude = tf.math.log1p(mu * safe_audio_abs) / tf.math.log1p(mu)
    # sign(x) log_e(1 + mu * min(|x|, 1)) / log_e(1 + mu)
    signal = tf.math.sign(audio) * magnitude
    return tf.cast((signal + 1) / 2 * mu + 0.5, dtype=tf.int32)


def mu_law_decode(output, quantization_channels=256):

    mu = tf.constant(quantization_channels - 1, dtype=tf.float32)
    signal = 2 * (tf.cast(output, dtype=tf.float32) / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** tf.math.abs(signal) - 1)
    return tf.sign(signal) * magnitude


class myThread(Thread):

    def __init__(self, file, writer, receptive_field, sample_rate,
                 silence_threshold, quantization_channels, pbar):
        Thread.__init__(self)
        self.file = file
        self.writer = writer
        self.receptive_field = receptive_field
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.quantization_channels = quantization_channels
        self.pbar = pbar

    def run(self):

        process_audio(self.file, self.writer, self.receptive_field,
                      self.sample_rate, self.silence_threshold, self.quantization_channels)
        self.pbar.update(1)


def process_audio(file, writer, receptive_field, sample_rate,
                  silence_threshold, quantization_channels):

    audio_path = file[0]
    audio, _ = loadAudio(audio_path, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)

    label_path = file[1]
    label = open(label_path, 'r')

    if label is None:

        print("Can't Open Label File " + label_path)
        return

    transcript = label.read().strip()
    label.close()
    class_id = int(file[2])

    # Trim silence under specific signal to noise ratio

    frame_length = 2048 if audio.size >= 2048 else audio.size
    energe = rms(audio, frame_length=frame_length)
    frames = np.nonzero(energe > silence_threshold)
    indices = frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
    audio = audio.reshape(-1, 1)

    # Pad at head

    audio = np.pad(audio, [[receptive_field, 0], [0, 0]], 'constant')

    # Quantization
    # quantized.shape = (length)
    quantized = mu_law_encode(audio, quantization_channels)

    trainsample = tf.train.Example(features=tf.train.Features(
        feature={
            'audio': tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(quantized, (-1,)))),
            'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[quantized.shape[0]])),
            'category': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_id])),
            'transcript': tf.train.Feature(bytes_list=tf.train.BytesList(value=[transcript.encode('utf-8')]))
        }
    ))
    writer.write(trainsample.SerializeToString())


def main(root_dir, args, dilations=[2 ** i for i in range(10)] * 5):

    receptive_field = calculate_receptive_field(dilations, 2, 32)
    category = {}  # person_id -> class id
    count = 0
    audiolist = []

    print("Building File Correlation Tree")

    for dir in listdir(join(root_dir, 'wav48')):
        for file in listdir(join(root_dir, 'wav48', dir)):

            result = search(r'p([0-9]+)_([0-9]+)\.wav', file)

            if result is None or not exists(join(root_dir, 'txt', dir, splitext(file)[0] + ".txt")):
                continue

            if result[1] not in category:
                category[result[1]] = count
                count += 1

            audiolist.append((join(root_dir, 'wav48', dir, file), join(
                root_dir, 'txt', dir, splitext(file)[0] + ".txt"), category[result[1]]))

    shuffle(audiolist)

    if not exists('dataset'):
        mkdir('dataset')

    writer = tf.io.TFRecordWriter(join('dataset', 'trainset.tfrecord'))

    print("Writing Tensorflow Dataset")

    if args['threading']:

        threads = []
        with tqdm(total=len(audiolist)) as pbar:
            for file in audiolist:
                thread = myThread(file, writer, receptive_field,
                                  args['sample_rate'], args['silence_threshold'], args['quantization_channels'], pbar)
                thread.start()
                threads.append(thread)

        for thread in threads:
            thread.join()

    else:

        for file in tqdm(audiolist):
            process_audio(file, writer, receptive_field,
                          args['sample_rate'], args['silence_threshold'], args['quantization_channels'])

    writer.close()
    category = [(class_id, person_id)
                for person_id, class_id in category.items()]
    category = pd.DataFrame(category, columns=['class_id', 'person_id'])
    category.to_pickle('category.pkl')
    category.to_excel('category.xls')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="path to YAML config file", default="./config.yaml")
    parser.add_argument(
        "-d", "--data", help="path to directory holding data files", default="./VCTK-Corpus")
    args = parser.parse_args()

    with open(args.config, 'r') as stream:

        main(args.data, loadYAML(stream))
