import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
torch.cuda.current_device()
#from preprocess import TorchOLAfrom AudioData import TrainingDataset, TrainCollate, EvalCollate, EvalDataset
from torch.utils.data import DataLoader

from AudioData import TrainingDataset, TrainCollate, EvalDataset, EvalCollate
from new_model import Net
#from STOI import stoi
#from PESQ import get_pesq
from metric import get_pesq, get_stoi
from helper_funcs import numParams, compLossMask, snr
from criteria import mse_loss, stftm_loss
from checkpoints import Checkpoint
import line_profiler

# import megengine as mge
# mge.dtr.eviction_threshold = "5GB" # 设置显存阈值为 5GB
# mge.dtr.enable() # 开启 DTR 显存优化


#设置选用哪块GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# hyperparameter
frame_size = 512
overlap = 0.5
frame_shift = int(512 * (1 - overlap))
#训练轮数
max_epochs = 150
#批大小 每次训练的样本个数
batch_size = 4
#初始学习速率
lr_init = 64 ** (-0.5)
eval_steps = 500
#权值衰减
weight_delay = 1e-7

# lr scheduling
step_num = 0
#通过warm_ups改变lr
warm_ups = 4000
#resampled重新采样
sr = 16000
resume_model = "/media/npu/Data/xhj/model/v9/latest.model-88.model" #不是None的话 就是相应的存model的路径

#model_save_path = '/media/concordia/DATA/KaiWang/pytorch_learn/pytorch_for_speech/voice_bank/Transformer/v5/checkpoints/transformer_light_0.4time.model'
model_save_path = "/media/npu/Data/xhj/model/v9/"
if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

early_stop = True

# file path
#train_file_list_path = '/media/concordia/DATA/KaiWang/pytorch_learn/pytorch_for_speech/voice_bank/Transformer/v1/train_file_list'
train_file_list_path = "/home/xhj/src/train_file_list"
validation_file_list_path = "/home/xhj/src/validation_file_list"

# data and data_loader
# num_workers = 4
train_data = TrainingDataset(train_file_list_path, frame_size=512, frame_shift=256)
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=8,
                          collate_fn=TrainCollate(),
                          pin_memory=True)

validation_data = EvalDataset(validation_file_list_path, frame_size=512, frame_shift=256)
validation_loader = DataLoader(validation_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=8,
                               collate_fn=EvalCollate(),
                               pin_memory=True)

# define model
model = Net()
#device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
#if torch.cuda.device_count() > 1:
#    print('Lets use', torch.cuda.device_count(), 'GPUs!')
#    net = torch.nn.DataParallel(model)
model = model.cuda()
print('Number of learnable parameters: %d' % numParams(model)) #可学习参数的数量

optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_delay)
#lr_list = [0.0002]*3 + [0.0001]*6 + [0.00005]*3 + [0.00001]*3


time_loss = mse_loss() #时域损失
freq_loss = stftm_loss() #频域损失


def validate(net, eval_loader, test_metric=False):
    net.eval()
    if test_metric:
        print('********Starting metrics evaluation on val dataset**********')
        total_stoi = 0.0
        total_snr = 0.0
        total_pesq = 0.0

    with torch.no_grad():
        count, total_eval_loss = 0, 0.0
        for k, (features, labels) in enumerate(eval_loader):
            features = features.cuda()  # [1, 1, num_frames, frame_size]
            labels = labels.cuda()  # [signal_len, ]

            output = net(features)  # [1, 1, sig_len_recover]
            output = output.squeeze()  # [sig_len_recover,]

            output = output[:labels.shape[-1]]  # keep length same (output label)

            eval_loss = torch.mean((output - labels) ** 2)
            total_eval_loss += eval_loss.data.item()

            est_sp = output.cpu().numpy()
            cln_raw = labels.cpu().numpy()
            if test_metric:
                st = get_stoi(cln_raw, est_sp, sr)
                pe = get_pesq(cln_raw, est_sp, sr)
                sn = snr(cln_raw, est_sp)
                total_pesq += pe
                total_snr += sn
                total_stoi += st

            count += 1
        avg_eval_loss = total_eval_loss / count
    net.train()
    if test_metric:
        return avg_eval_loss, total_stoi / count, total_pesq / count, total_snr / count
    else:
        return avg_eval_loss



# train model
if resume_model:
    print('Resume model from "%s"' % resume_model)  #存储model的路径
    checkpoint = Checkpoint()
    checkpoint.load(resume_model)

    start_epoch = checkpoint.start_epoch + 1
    best_val_loss = checkpoint.best_val_loss
    prev_val_loss = checkpoint.prev_val_loss
    num_no_improv = checkpoint.num_no_improv
    half_lr = checkpoint.half_lr
    model.load_state_dict(checkpoint.state_dict)
    optimizer.load_state_dict(checkpoint.optimizer)

else:
    print('Training from scratch.') #从零开始训练
    start_epoch = 0
    best_val_loss = float("inf")
    prev_val_loss = float("inf")
    num_no_improv = 0
    half_lr = False

for epoch in range(start_epoch, max_epochs):
    model.train()
    total_train_loss, count, ave_train_loss = 0.0, 0, 0.0

    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_list[epoch]
    '''

    '''
    if half_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2
            print('Learning rate adjusted to  %5f' % (param_group['lr']))
        half_lr = False
    '''
    #根据论文中的描述调整学习速率
    for index, (features, labels, sig_len) in enumerate(train_loader):

        step_num += 1
        if step_num <= warm_ups:
            lr = 0.2 * lr_init * min(step_num ** (-0.5),
                                     step_num * (warm_ups ** (-1.5)))
        else:
            lr = 0.0004 * (0.98 ** ((epoch - 1) // 2))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            #print('Learning rate adjusted to  %6f' % (param_group['lr']))


        # feature -- [batch_size, 1, nframes, frame_size]
        features = features.cuda()
        # label -- [batch_size, 1, signal_length]
        labels = labels.cuda()

        loss_mask = compLossMask(labels, nframes=sig_len)

        optimizer.zero_grad()
# 釋放內存
        #with torch.no_grad():
        output = model(features)  # output -- [batch_size, 1, sig_len_recover]


        output = output[:, :, :labels.shape[-1]]  # [batch_size, 1, sig_len]


        loss_time = time_loss(output, labels, loss_mask)
        loss_freq = freq_loss(output, labels, loss_mask)

        # loss = 0.4 * loss_time + 0.6 * loss_freq
        loss = 0.2 * loss_time + 0.8 * loss_freq
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        train_loss = loss.data.item()
        total_train_loss += train_loss

        count += 1

        del loss, loss_time, loss_freq, output, loss_mask, features, labels
        #print('iter = {}/{}, epoch = {}/{}, loss = {:.5f}'.format(index + 1, len(train_loader), epoch + 1, max_epochs, train_loss))

        # if (index + 1) % eval_steps == 0:
        #     ave_train_loss = total_train_loss / count
        # 
        #     # validation
        #     avg_eval_loss = validate(model, validation_loader)
        #     model.train()
        # 
        #     print('Epoch [%d/%d], Iter [%d/%d],  ( TrainLoss: %.4f | EvalLoss: %.4f )' % (
        #     epoch + 1, max_epochs, index + 1, len(train_loader), ave_train_loss, avg_eval_loss))
        # 
        #     count = 0
        #     total_train_loss = 0.0


        if (index + 1) % len(train_loader) == 0:
            break

    # validate metric
    avg_eval, avg_stoi, avg_pesq, avg_snr = validate(model, validation_loader, test_metric=True)
    model.train()
    print('#' * 50)
    print('')
    print('After {} epoch the performance on validation score is a s follows:'.format(epoch + 1))
    print('')
    print('Avg_loss: {:.4f}'.format(avg_eval))
    print('STOI: {:.4f}'.format(avg_stoi))
    print('SNR: {:.4f}'.format(avg_snr))
    print('PESQ: {:.4f}'.format(avg_pesq))


    # adjust learning rate and early stop
    if avg_eval >= prev_val_loss:
        num_no_improv += 1
        #if num_no_improv == 2:
            #half_lr = True
        if num_no_improv >= 10 and early_stop is True:
            print("No improvement and apply early stop")
            break
    else:
        num_no_improv = 0

    prev_val_loss = avg_eval

    if avg_eval < best_val_loss:
        best_val_loss = avg_eval
        is_best_model = True
    else:
        is_best_model = False

    # save model
    latest_model = 'latest.model'
    best_model = 'best.model'

    checkpoint = Checkpoint(start_epoch=epoch,
                            best_val_loss=best_val_loss,
                            prev_val_loss=prev_val_loss,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            num_no_improv=num_no_improv,
                            half_lr=half_lr)
    checkpoint.save(is_best=is_best_model,
                    filename=os.path.join(model_save_path, latest_model + '-{}.model'.format(epoch + 1)),
                    best_model=os.path.join(model_save_path, best_model))


