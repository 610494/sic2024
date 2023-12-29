from utils.generate import generate_audio
from model.nn import SimpleNN, CustomDataset
import soundfile as sf

import torch
from torch import nn, utils
from torch.utils.data import random_split, DataLoader, Dataset

import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor

import pandas as pd
import torch

# sr = 16000
# audios = generate_audio(number=5, sr=sr)
# for index, audio in enumerate(audios):
#     sf.write(f'audio_{index}.wav', audio, sr)


def data_train(x1, x2, y, is_test=False):
    # 定义要分割的比例
    train_ratio = 0.7
    number = len(x1)

    # 计算要分割的数量
    train_size = int(train_ratio * number)
    if is_test:
        valid_size = (number - train_size) // 2
        test_size = number - train_size - valid_size
    else:
        valid_size = (number - train_size)
        test_size = 0

    # 使用 random_split 分割数据集
    train_x1, valid_x1, test_x1 = torch.split(
        x1, [train_size, valid_size, test_size])
    train_x2, valid_x2, test_x2 = torch.split(
        x2, [train_size, valid_size, test_size])
    train_y, valid_y, test_y = torch.split(
        y, [train_size, valid_size, test_size])

    return CustomDataset(train_x1, train_x2, train_y), CustomDataset(valid_x1, valid_x2, valid_y), CustomDataset(test_x1, test_x2, test_y)


lr_monitor_callback = LearningRateMonitor(logging_interval='step')

ckpt_callback = pl.pytorch.callbacks.ModelCheckpoint(
    dirpath=f'paraNN-cp',
    filename="paraNN-{epoch}-{step}-{val_loss:.2f}",
    monitor="val_loss",
    save_top_k=10,
    every_n_epochs=1,
)

if __name__ == '__main__':
    # number = 100
    # # 创建训练数据
    # x1 = torch.randn(number, 4)  # 随机生成两个输入数据，100 组数据
    # x2 = torch.randn(number, 4)
    # y = torch.randn(number, 4)   # 随机生成对应的输出数据

    df = pd.read_csv('data/nn_data_4000.csv')

    # 将每列数据转换为张量
    tensor_dict = {}

    before_list = ['before_sig_MOS_DISC', 'before_sig_MOS_LOUD',
                   'before_sig_MOS_NOISE', 'before_sig_MOS_REVERB']
    after_list = ['after_sig_MOS_DISC', 'after_sig_MOS_LOUD',
                  'after_sig_MOS_NOISE', 'after_sig_MOS_REVERB']
    para_list = ['para_sig_MOS_DISC', 'para_sig_MOS_LOUD',
                 'para_sig_MOS_NOISE', 'para_sig_MOS_REVERB']

    for column in before_list+after_list+para_list:
        tensor_dict[column] = torch.tensor(df[column].values).float()
        if column in before_list+after_list:
            mean = tensor_dict[column].mean()
            std = tensor_dict[column].std()
            normalized_data = (tensor_dict[column] - mean) / std
            tensor_dict[column] = torch.tensor(normalized_data)

    before_mos = torch.cat(tuple(torch.tensor(i) for i in zip(
        *[tensor_dict[j] for j in before_list])), dim=0)
    after_mos = torch.cat(tuple(torch.tensor(i) for i in zip(
        *[tensor_dict[j] for j in after_list])), dim=0)
    para_mos = torch.cat(tuple(torch.tensor(i) for i in zip(
        *[tensor_dict[j] for j in para_list])), dim=0)

    before_mos = before_mos.view(before_mos.numel() // 4, 4)
    after_mos = after_mos.view(after_mos.numel() // 4, 4)
    para_mos = para_mos.view(para_mos.numel() // 4, 4)

    training_set, valid_set, test_set = data_train(
        before_mos, after_mos, para_mos)

    training_loader = DataLoader(training_set, batch_size=8, shuffle=True)
    vaild_loader = DataLoader(valid_set, batch_size=8, shuffle=False)

    model = SimpleNN(lr_sch=False)
    trainer = pl.Trainer(max_epochs=3000,
                         accelerator="gpu",
                         devices=1,
                         strategy="ddp",
                         enable_checkpointing=True,
                         check_val_every_n_epoch=1,
                         callbacks=[ckpt_callback, lr_monitor_callback])
    trainer.fit(model, training_loader, vaild_loader)

    # all_lose = 0
    # for batch in test_set:
    #     x1, x2, y = batch
    #     y_pred = model.inference(x1, x2)

    #     all_lose += nn.MSELoss()(y_pred, y)

    # print(f'test MSE loss: {all_lose/len(test_set)}')
