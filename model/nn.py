import torch
from torch import nn, utils
from torch.utils.data import Dataset, DataLoader

import lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


class CustomDataset(Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        # self.x1 = x1.view(-1, batch_size, x1.shape[1])
        # self.x2 = x2.view(-1, batch_size, x2.shape[1])
        # self.y = y.view(-1, batch_size, y.shape[1])

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]


class SimpleNN(pl.LightningModule):
    def __init__(self, input_dim=8, hidden_dims=[8, 6, 4], output_dim=4, dropout_prob=0.5, lr_sch=False):
        super(SimpleNN, self).__init__()

        all_dims = [input_dim] + hidden_dims + [output_dim]
        layers = []

        # 创建隐藏层和输出层
        for i in range(len(all_dims) - 2):
            layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            if i < len(all_dims) - 1 and i > 0:
                layers.append(nn.ReLU())  # 隐藏层使用 ReLU 激活函数
                # layers.append(nn.Dropout(dropout_prob))  # 在隐藏层后添加 Dropout 正则化

        # 将层组合成一个顺序的神经网络
        self.layers = nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
        self.lr_sch = lr_sch
        if lr_sch:
            self.scheduler = StepLR(self.optimizer, step_size=150, gamma=0.1)
            # self.scheduler = ReduceLROnPlateau(
            #     self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
            # self.validation_step_outputs = []

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch):
        x1, x2, y = batch
        b, _ = x1.shape
        loss = 0
        for i in range(b):
            y_pred = self(torch.cat((x1[i], x2[i]), dim=0))
            loss += nn.L1Loss()(y_pred, y[i])

        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch):
        x1, x2, y = batch
        b, _ = x1.shape
        val_loss = 0
        for i in range(b):
            y_pred = self(torch.cat((x1[i], x2[i]), dim=0))
            val_loss += nn.L1Loss()(y_pred, y[i])
        # self.log('val_loss', val_loss)
        self.log('val_loss', val_loss, sync_dist=True)

        # if self.lr_sch:
        #     self.validation_step_outputs.append(val_loss)
        return val_loss

    # def on_validation_epoch_end(self):
    #     if self.lr_sch:
    #         epoch_average = torch.stack(self.validation_step_outputs).mean()
    #         self.validation_step_outputs.clear()
    #         self.scheduler.step(epoch_average)

        # # 返回验证指标（在 Lightning 中，你可以通过此方法返回想要的任何指标）
        # return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        if self.lr_sch:
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': {
                    'scheduler': self.scheduler,
                    'interval': 'epoch',
                }
            }
        else:
            return self.optimizer

    def inference(self, x1, x2):
        y_pred = self.layers(torch.cat((x1, x2), dim=0))
        return y_pred

    def load_checkpoint(self, checkpoint_path):
        # 加载检查点文件
        spec = torch.load(checkpoint_path)

        # spec = torch.load(local_filename, map_location='cpu')
        assert 'pytorch-lightning_version' in spec, 'not a valid PyTorch Lightning checkpoint'
        state_dict = {k.replace('model.', ''): v
                      for k, v in spec['state_dict'].items()}
        self.load_state_dict(state_dict)
        return self
        # # 加载模型状态
        # self.load_state_dict(checkpoint['model_state_dict'])

        # # 如果启用了 lr_scheduler，则加载优化器和调度器状态
        # if self.lr_sch:
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # # 如果要加载 epoch 数和其他信息，可以像下面这样操作
        # self.current_epoch = checkpoint['epoch']
        # self.some_other_info = checkpoint['some_other_info']

        # print("Checkpoint loaded successfully.")

# # 创建训练数据
# x1 = torch.randn(100, 4, 3)  # 随机生成两个输入数据，100 组数据
# # x2 = torch.randn(100, 4)
# # y = torch.randn(100, 4)   # 随机生成对应的输出数据

# # 创建 Lightning Trainer 并训练模型
# model = SimpleNN()
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, utils.data.DataLoader(x1))
