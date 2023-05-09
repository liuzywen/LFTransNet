import torch
import torch.nn.functional as F
import sys


sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from data import test_dataset
from model.LFTransNet import model
from torchvision.utils import save_image


print("GPU available:", torch.cuda.is_available())


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = model()
model.load_state_dict(torch.load('/'))

# test
model.cuda()
model.eval()


def CAM(features, img_path, save_path):
    features.retain_grad()
    # t = model.avgpool(features)
    # t = t.reshape(1, -1)
    # output = model.classifier(t)[0]

    # pred = torch.argmax(output).item()
    # pred_class = output[pred]
    #
    # pred_class.backward()
    grads = features.grad
    # features = torch.cat(torch.chunk(features, 12, dim=0), dim=1)[0]
    features = features.squeeze(0)
    # print(features.shape)
    # avg_grads = torch.mean(grads[0], dim=(1, 2))
    # avg_grads = avg_grads.expand(features.shape[1], features.shape[2], features.shape[0]).permute(2, 0, 1)
    # features *= avg_grads

    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    # print(img_path)
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
    # cv2.imshow('1', superimposed_img)
    # cv2.waitKey(0)
    cv2.imwrite(save_path, superimposed_img)




test_datasets = ['DUTLF-FS','HFUT','LFSD']
for dataset in test_datasets:
    save_path = './test_maps/' + dataset + '/'
    save_path1 = './test_maps1/' + dataset + '/'
    save_path2 = './test_maps2/' + dataset + '/'
    save_path3 = './test_maps3/' + dataset + '/'
    save_path4 = './test_maps4/' + dataset + '/'
    save_path_sde = './sde_maps/' + dataset + '/'
    save_path_decoder = './decoder_maps/' + dataset + '/'
    # save_path_lastConv = './lastConv_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path1):
        os.makedirs(save_path1)

    if not os.path.exists(save_path2):
        os.makedirs(save_path2)

    if not os.path.exists(save_path3):
        os.makedirs(save_path3)

    if not os.path.exists(save_path4):
        os.makedirs(save_path4)

    if not os.path.exists(save_path_sde):
        os.makedirs(save_path_sde)

    if not os.path.exists(save_path_decoder):
        os.makedirs(save_path_decoder)

    image_root = dataset_path + dataset + '/test_images/'
    gt_root = dataset_path + dataset + '/test_masks/'
    fs_root = dataset_path + dataset + '/test_focals/'
    test_loader = test_dataset(image_root, gt_root, fs_root, opt.testsize)
    for i in range(test_loader.size):
        #todo 位置
        image, focal, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        dim, height, width = focal.size()
        basize = 1
        focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
        focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
        focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
        focal = focal.view(-1, *focal.shape[2:])  # [basize*12, 6, 256, 256)
        focal = focal.cuda()
        image = image.cuda()
        s1, s2, s3, s4, res, xf, xq, sde, fuse_sal = model(focal, image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res * 255)


        img_path = image_root+name
        img_path = img_path.split('.')[0]+'.jpg'
        # CAM(xf, img_path, save_path1 + name)
        # CAM(xf, img_path, save_path3 + name)
        # CAM(xq, img_path, save_path2 + name)
        CAM(sde, img_path, save_path_sde + name)
        CAM(fuse_sal, img_path, save_path_decoder + name)



        # print(xf.shape, xq.shape)  # torch.Size([12, 32, 64, 64]) torch.Size([12, 32, 64, 64])
        # xf = xf[0, 0, :, :].unsqueeze(0)
        # xq = xq[0, :, :, :].unsqueeze(0)
        # print(xf.shape)
        # save_image(xf, save_path1+name)
        # save_image(xq, save_path2+name)
        # # s1 = F.upsample(s1, size=(64, 64), mode='bilinear', align_corners=False)
        # s1 = s1.sigmoid().data.cpu().numpy().squeeze()
        # s1 = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-8)
        # print('save img to: ', save_path + name)
        # cv2.imwrite(save_path1 + name, s1 * 255)
        #
        # s2 = F.upsample(s2, size=(64, 64), mode='bilinear', align_corners=False)
        # s2 = s2.sigmoid().data.cpu().numpy().squeeze()
        # s2 = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-8)
        # print('save img to: ', save_path + name)
        # cv2.imwrite(save_path2 + name, s2 * 255)
        #
        # s3 = F.upsample(s3, size=(64, 64), mode='bilinear', align_corners=False)
        # s3 = s3.sigmoid().data.cpu().numpy().squeeze()
        # s3 = (s3 - s3.min()) / (s3.max() - s3.min() + 1e-8)
        # print('save img to: ', save_path + name)
        # cv2.imwrite(save_path3 + name, s3 * 255)
        #
        # s4 = F.upsample(s4, size=(64, 64), mode='bilinear', align_corners=False)
        # s4 = s4.sigmoid().data.cpu().numpy().squeeze()
        # s4 = (s4 - s4.min()) / (s4.max() - s4.min() + 1e-8)
        # print('save img to: ', save_path + name)
        # cv2.imwrite(save_path4 + name, s4 * 255)

    print('Test Done!')
