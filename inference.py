from model.nn import SimpleNN
import torch

if __name__ == '__main__':
    cp_file = '/share/nas165/sic2024/datasets/SIG-Challenge/ICASSP2024/nn/paraNN-cp/4000-[16,16,8,8,4]_unnor/paraNN-epoch=1113-step=389900-val_loss=18714.72.ckpt'

    model = SimpleNN()
    model.load_checkpoint(cp_file)

    random_tensor = torch.randint(0, 500, (1, 8))
    random_tensor = random_tensor.to(torch.float32)
    y_pred = model.inference(random_tensor[:4], random_tensor[4:])

    print(f'tensor: {random_tensor}, y_pred: {y_pred}')
