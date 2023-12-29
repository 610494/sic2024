import os
import json
import pandas as pd
import librosa
from tqdm import tqdm

from sigmos.sigmos import SigMOS


def data_prepare(audios_path, before_mos_path, para_path, delta=0.002, output='output.csv'):
    # store para
    para_list = [None] * 100
    mos_list = ['sig_MOS_DISC', 'sig_MOS_LOUD', 'sig_MOS_NOISE',
                'sig_MOS_REVERB', 'sig_MOS_SIG', 'sig_MOS_OVRL']

    before_mos_list = ['before_' + item for item in mos_list]
    after_mos_list = ['after_' + item for item in mos_list]
    para_mos_list = ['para_' + item for item in mos_list]
    df = pd.DataFrame(columns=['filename'] +
                      before_mos_list+after_mos_list+para_mos_list)

    # 遍歷目錄中的所有檔案
    for filename in os.listdir(para_path):
        if filename.endswith('.json'):
            # 解析檔名以獲取數字部分
            parts = filename.split('.')[0].split('_')
            if len(parts) == 3 and parts[2].isdigit():
                index = int(parts[2])
                # 讀取JSON檔案並存儲到list的相應index中
                with open(os.path.join(para_path, filename), 'r') as file:
                    json_data = json.load(file)
                    para_list[index] = json_data

    before_mos_df = pd.read_csv(before_mos_path)

    for mos_name in mos_list:
        before_mos_df[mos_name] = before_mos_df[mos_name] // (5*delta)

    index = 0
    # deal_audios
    model_dir = r"/share/nas165/sic2024/datasets/SIG-Challenge/ICASSP2024/sigmos"
    sigmos_estimator = SigMOS(model_dir=model_dir)
    for root, dirs, files in os.walk(audios_path):
        for file in tqdm(files):
            if file.endswith('.wav'):
                audio, sr = librosa.load(os.path.join(
                    root, file), sr=None)  # sr=None 表示保持原始采样率
                all_mos = sigmos_estimator.run(audio, sr=sr)
                syn_mos = []
                for mos_name in mos_list:
                    syn_mos.append(int(all_mos[mos_name] // (5*delta)))

                para = int(file.split('_')[0])
                filename = file.split('_', 1)[1]

                if para >= 40:
                    continue

                newrow = [file]

                before_mos = before_mos_df[before_mos_df['deg'] == filename]
                for mos in mos_list:
                    newrow.append(int(before_mos[mos].values))

                newrow = newrow + syn_mos

                # para_mapping = {'sig_MOS_DISC': 'disc', 'sig_MOS_REVERB': 'reverb',
                #                 'sig_MOS_NOISE': 'noise', 'sig_MOS_LOUD': 'gain'}

                para_key_mapping = {'sig_MOS_DISC': 'disc', 'sig_MOS_REVERB': 'reverb',
                                    'sig_MOS_NOISE': 'noise', 'sig_MOS_LOUD': 'gain'}
                para_value_mapping = {'sig_MOS_DISC': 'db', 'sig_MOS_REVERB': 'room_size',
                                      'sig_MOS_NOISE': 'db', 'sig_MOS_LOUD': 'db'}
                for mos in mos_list:
                    if mos in ['sig_MOS_SIG', 'sig_MOS_OVRL']:
                        newrow.append(0)
                    else:
                        newrow.append(para_list[para][para_key_mapping[mos]
                                                      ][int(before_mos[mos].values[0])][para_value_mapping[mos]][int(all_mos[mos] // (5*delta))])

                df.loc[len(df)] = newrow

                # if index >= 3:
                #     break

                # index += 1
                df.to_csv(output, index=False)


if __name__ == '__main__':
    audios_path = '/share/nas165/sic2024/datasets/SIG-Challenge/ICASSP2024/test_output/all_para_data'
    before_mos_path = '/share/nas165/sic2024/datasets/SIG-Challenge/ICASSP2024/ana_data_per_file_test_data.csv'
    para_path = '/share/nas165/sic2024/datasets/SIG-Challenge/ICASSP2024/parameter'
    data_prepare(audios_path, before_mos_path, para_path,
                 delta=0.002, output='new_nn_data.csv')
