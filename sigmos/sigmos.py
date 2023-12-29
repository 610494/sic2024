import os
import scipy
import librosa
import fnmatch
from tqdm import tqdm

import numpy as np
import onnxruntime as ort
from enum import Enum
import csv
import pandas as pd


__all__ = ["SigMOS", "Version"]


class Version(Enum):
    V1 = "v1"  # 15.10.2023


class SigMOS:
    '''
    MOS Estimator for the P.804 standard.
    See https://arxiv.org/pdf/2309.07385.pdf
    '''

    def __init__(self, model_dir, model_version=Version.V1):
        assert model_version in [v for v in Version]

        model_path_history = {
            Version.V1: os.path.join(
                model_dir, 'model-sigmos_1697718653_41d092e8-epo-200.onnx')
        }

        self.sampling_rate = 48_000
        self.resample_type = 'fft'
        self.model_version = model_version

        # STFT params
        self.dft_size = 960
        self.frame_size = 480
        self.window_length = 960
        self.window = np.sqrt(np.hanning(
            int(self.window_length) + 1)[:-1]).astype(np.float32)

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path_history[model_version], options)

    def stft(self, signal):
        last_frame = len(signal) % self.frame_size
        if last_frame == 0:
            last_frame = self.frame_size

        padded_signal = np.pad(
            signal, ((self.window_length - self.frame_size, self.window_length - last_frame),))
        frames = librosa.util.frame(padded_signal, frame_length=len(
            self.window), hop_length=self.frame_size, axis=0)
        spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
        return spec.astype(np.complex64)

    @staticmethod
    def compressed_mag_complex(x: np.ndarray, compress_factor=0.3):
        x = x.view(np.float32).reshape(x.shape + (2,)).swapaxes(-1, -2)
        x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
        if compress_factor == 1:
            mag = np.sqrt(x2)
        else:
            x = np.power(x2, (compress_factor - 1) / 2) * x
            mag = np.power(x2, compress_factor / 2)

        features = np.concatenate((mag, x), axis=-2)
        features = np.transpose(features, (1, 0, 2))
        return np.expand_dims(features, 0)

    def run(self, audio: np.ndarray, sr=None):
        if sr is not None and sr != self.sampling_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.sampling_rate, res_type=self.resample_type)
            print(f"Audio file resampled from {sr} to {self.sampling_rate}!")

        features = self.stft(audio)
        features = self.compressed_mag_complex(features)

        onnx_inputs = {inp.name: features for inp in self.session.get_inputs()}
        output = self.session.run(None, onnx_inputs)[0][0]

        result = {
            'sig_MOS_COL': float(output[0]), 'sig_MOS_DISC': float(output[1]), 'sig_MOS_LOUD': float(output[2]),
            'sig_MOS_NOISE': float(output[3]), 'sig_MOS_REVERB': float(output[4]), 'sig_MOS_SIG': float(output[5]),
            'sig_MOS_OVRL': float(output[6])
        }
        return result


def save_dictlist_to_csv(data_list, file_name):
    if not data_list:
        print("The data list is empty. Nothing to save.")
        return

    # 提取所有字典的键以确定CSV文件的标题
    fieldnames = data_list[0].keys()

    try:
        with open(file_name, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()  # 写入CSV文件标题行

            for data in data_list:
                writer.writerow(data)  # 逐行写入数据

        print(f"Data saved to {file_name} successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# 定义一个函数来获取目录中的所有.wav文件


def find_wav_files(directory):
    wav_files = []
    for root, dirs, files in tqdm(os.walk(directory)):
        print(root)
        for filename in files:
            if filename.endswith('.wav'):
                wav_files.append(os.path.join(root, filename))
    return wav_files

# # 调用函数获取所有.wav文件的列表
# wav_file_list = find_wav_files(directory)

# # 打印所有找到的.wav文件
# for wav_file in wav_file_list:


def ana(audio_dir):
    result_list = []
    audio_list = find_files(audio_dir)
    result

    for audio_file_path in tqdm(audio_list):
        audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
        dummy_result = sigmos_estimator.run(audio_data, sr=sample_rate)
        dummy_result['FILE_NAME'] = audio_file_path[audio_file_path.rfind(
            '/') + 1:]
        result_list.append(dummy_result)
    return result_list


def ana2csv(audio_dir):
    result_list = ana(audio_dir)
    csv_name = audio_dir[audio_dir.rfind('/') + 1:] + '.csv'

    save_dictlist_to_csv(result_list, csv_name)
    return csv_name


if __name__ == '__main__':
    ''' 
        Sample code to run the SigMOS estimator. 
        V1 (current model) is an alpha version and should be used in accordance.
    '''
    model_dir = r"."
    sigmos_estimator = SigMOS(model_dir=model_dir)

    # # input data must have sr=48kHz, otherwise please specify the sr (it will be resampled to 48kHz internally)
    # sampling_rate = 48_000
    # dummy_data = np.random.rand(5 * sampling_rate)
    # dummy_result = sigmos_estimator.run(dummy_data, sr=sampling_rate)
    # print(dummy_result)

    audio_dir = '/share/nas165/sic2024/datasets/DNS-Challenge/datasets_fullband/datasets_fullband/clean_fullband'
    # audio_dir = '/share/nas165/sic2024/datasets/SIG-Challenge/ICASSP2024/demo_synthesizer/result/test'

    print('read files')
    wav_files = find_wav_files(audio_dir)

    sampling_rate = 48_000

    print('count sigmos')
    result = pd.DataFrame()
    for wav_file in tqdm(wav_files):
        # print('wav_file:', wav_file)
        audio, _ = librosa.load(wav_file, sr=sampling_rate)
        new_row = sigmos_estimator.run(audio, sr=sampling_rate)
        result = pd.concat(
            [result, pd.DataFrame([new_row])], ignore_index=True)

    result.to_csv('result.csv', index=False)
    # print(audio_dir[audio_dir.rfind('/') + 1:])
