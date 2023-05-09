import logging
import os
from datetime import datetime
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
# from torch.utils.tensorboard import SummaryWriter
from data import get_loader, test_dataset
from model.LFTransNet import model
from options import opt
from utils import clip_gradient, adjust_lr
from tools.pytorch_utils import Save_Handle
from torch.autograd import Variable

torch.cuda.current_device()

print("GPU available:", torch.cuda.is_available())

if opt.gpu_id == '0':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('USE GPU 1')
cudnn.benchmark = True

save_list = Save_Handle(max_num=1)


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()#numel用于返回数组中的元素个数
    print(name)
    print('The number of parameters:{}'.format(num_params))
    return num_params
start_epoch = 0

model = model()
if (opt.load_mit is not None):
    model.focal_encoder.init_weights(opt.load_mit)
    model.rgb_encoder.init_weights(opt.load_mit)
else:
    print("No pre-trian!")

model.cuda()
params = model.parameters()
optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)  #weight_decay正则化系数
model_params = print_network(model, 'lf_pvt')
rgb_root = opt.rgb_root
gt_root = opt.gt_root
fs_root = opt.fs_root
test_rgb_root = opt.test_rgb_root
test_fs_root = opt.test_fs_root
test_gt_root = opt.test_gt_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(rgb_root, gt_root, fs_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_rgb_root, test_gt_root, test_fs_root,testsize=opt.trainsize)
total_step = len(train_loader)
logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Net-Train")
logging.info("Config")
logging.info('params:{}'.format(model_params))
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate,  save_path,
        opt.decay_epoch))


#set loss function
# CE = torch.nn.BCEWithLogitsLoss()


def structure_loss(pred, mask):
    weit = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


step = 0
# writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0



def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, focal) in enumerate(train_loader, start=1):
            basize, dim, height, width = focal.size()
            gts = gts.cuda()
            images, gts, focal = Variable(images), Variable(gts), Variable(focal)
            gts1 = F.interpolate(gts, size=(64, 64), mode='bilinear', align_corners=False)
            gts2 = F.interpolate(gts, size=(32, 32), mode='bilinear', align_corners=False)
            gts3 = F.interpolate(gts, size=(16, 16), mode='bilinear', align_corners=False)
            gts4 = F.interpolate(gts, size=(8, 8), mode='bilinear', align_corners=False)
            focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
            focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
            focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
            focal = focal.view(-1, *focal.shape[2:])  # [basize*12, 6, 256, 256)
            focal = focal.cuda()
            images = images.cuda()
            optimizer.zero_grad()
            x1, x2, x3, x4, focal_sal, _, _ = model(focal, images)
            loss = structure_loss(focal_sal, gts) + structure_loss(x1, gts1) + structure_loss(x2, gts2) + structure_loss(x3, gts3) + structure_loss(x4, gts4)
            loss.backward()
            # 梯度裁剪
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} ||Mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data, memory_used))
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'lfsod_epoch_{}.pth'.format(epoch))

        # 训练中断保留参数
        temp_save_path = save_path + "{}_ckpt.tar".format(epoch)
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict()
        }, temp_save_path)
        save_list.append(temp_save_path)

    except KeyboardInterrupt:
        logging.info('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'lfnet_epoch_{}.pth'.format(epoch + 1))
        logging.info('save checkpoints successfully!')
        raise





def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, focal, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            dim, height, width = focal.size()
            basize = 1


            focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
            focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
            focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
            focal = focal.view(-1, *focal.shape[2:])
            focal = focal.cuda()
            image = image.cuda()


            _, _, _, _, res = model(focal, image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        logging.info('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'lfsod_epoch_best.pth')
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    logging.info("Start train...")
    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    for epoch in range(start_epoch, opt.epoch+1):
        # if (epoch % 50 ==0 and epoch < 60):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)

        test(test_loader, model, epoch, save_path)





































