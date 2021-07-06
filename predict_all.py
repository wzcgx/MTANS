import argparse
import os
import shutil
import time
import logging
import random

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim
cudnn.benchmark = True
from scipy.stats import pearsonr
import numpy as np
from skimage import measure
from medpy import metric

import models
from models import criterions
from data import datasets
from data.data_utils_isles import add_mask
from utils import Parser

path = os.path.dirname(__file__)

# def calculate_metrics(pred, target):
    # sens = metric.sensitivity(pred, target)
    # spec = metric.specificity(pred, target)
    # dice = metric.dc(pred, target)

eps = 1e-5
def f1_score(o, t):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den

def ppv(o, t):    
    seg_total = np.sum(o)
    truth_total = np.sum(t)
    tp = np.sum(o[t == 1])
    ppv = tp / (seg_total + 0.001)
    return ppv

def tpr(o, t): 
    seg_total = np.sum(o)
    truth_total = np.sum(t)   
    tp = np.sum(o[t == 1])
    tpr = tp / (truth_total + 0.001)
    return tpr

def lfpr(o, t):    
    seg_labels, seg_num = measure.label(o, return_num=True, connectivity=2)
    lfp_cnt = 0
    tmp_cnt = 0
    for label in range(1, seg_num + 1):
        tmp_cnt += np.sum(o[seg_labels == label])
        if np.sum(t[seg_labels == label]) == 0:
            lfp_cnt += 1
    lfpr = lfp_cnt / (seg_num + 0.001)
    return lfpr  

def ltpr(o, t):    
    truth_labels, truth_num = measure.label(t, return_num=True, connectivity=2)
    ltp_cnt = 0
    for label in range(1, truth_num + 1):
        if np.sum(o[truth_labels == label]) > 0:
            ltp_cnt += 1
    ltpr = ltp_cnt / truth_num
    return ltpr  

def corr(o, t):    
    pearsonr(o.flatten(), t.flatten())[0]
    return corr    
    
def vd(o, t):    
    seg_total = np.sum(o)
    truth_total = np.sum(t)
    vd = abs(seg_total - truth_total) / truth_total    
    return vd  
#https://github.com/ellisdg/3DUnetCNN
#https://github.com/ellisdg/3DUnetCNN/blob/master/brats/evaluate.py
#https://github.com/MIC-DKFZ/BraTS2017
#https://github.com/MIC-DKFZ/BraTS2017/blob/master/utils_validation.py
def dice(output, target):
    ret = []
    # whole
    o = output > 0; t = target > 0
    ret += f1_score(o, t),

    print('f1_score',f1_score(o, t))
    # whole
    o = output > 0; t = target > 0
    ret += ppv(o, t),
    print('ppv(o, t)',ppv(o, t))
    # whole
    o = output > 0; t = target > 0
    ret += tpr(o, t),
    print('tpr(o, t)',tpr(o, t))    
    # whole
    o = output > 0; t = target > 0
    ret += lfpr(o, t),
    print('lfpr(o, t)', lfpr(o, t))   
    # whole
    o = output > 0; t = target > 0
    ret += ltpr(o, t),
    print('ltpr(o, t)', ltpr(o, t))      
    # whole
    o = output > 0; t = target > 0
    ret += vd(o, t),
    print('vd(o, t)', vd(o, t))      
     # # whole
    # o = output > 0; t = target > 0
    # ret += corr(o, t),   
    # print('corr(o, t)', corr(o, t))       
    
    # # whole
    # o = output > 0; t = target > 0

    # if 0 == np.count_nonzero(o):
        # score=100
    # else:
        # score = metric.hd(o , t)
    # ret += score,


    # # whole
    # o = output > 0; t = target > 0
    # if 0 == np.count_nonzero(o):
        # score=100
    # else:
        # score = metric.assd(o , t)
    # ret += score,

    return ret

keys = 'dice', 'ppv', 'tpr', 'lfpr', 'ltpr', 'vd'
def main():
    ckpts = args.getdir()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # setup networks
    Network = getattr(models, args.net)
    model = Network(**args.net_params)
    model = model.cuda()

    model_file = os.path.join(ckpts, args.ckpt)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])

    Dataset = getattr(datasets, args.dataset)

    valid_list = os.path.join(args.data_dir, args.valid_list)
    valid_set = Dataset(valid_list, root=args.data_dir,
            for_train=False, return_target=args.scoring,
            transforms=args.test_transforms)
    valid_loader = DataLoader(
        valid_set,
        batch_size=1, shuffle=False,
        collate_fn=valid_set.collate,
        num_workers=4, pin_memory=True)

    start = time.time()
    with torch.no_grad():
        scores = validate(valid_loader, model,
                args.out_dir, valid_set.names, scoring=args.scoring)

    msg = 'total time {:.4f} minutes'.format((time.time() - start)/60)
    logging.info(msg)


def validate(valid_loader, model,ckpt,ckpts_dir,
        out_dir='', names=None, scoring=True, verbose=True):
        
    model_file = os.path.join(ckpts_dir, ckpt)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])
    H, W, T = 181, 217, 181
    dtype = torch.float32

    dset = valid_loader.dataset

    model.eval()
    criterion = F.cross_entropy

    vals = AverageMeter()
    for i, data in enumerate(valid_loader):

        target_cpu = data[1][0, :H, :W, :T].numpy() if scoring else None
        data = [t.cuda(non_blocking=True) for t in data]

        x, target = data[:2]
        print('x_origin_size',x.shape)
        print('target',target.shape)
        if len(data) > 2:
            x = add_mask(x, data.pop(), 1)
        print('x_size',x.shape)
        # compute output
        # if(x.shape[4]==159):
            # print('pad to 160')
            # p1d = (0, 1)
            # x= F.pad(x, p1d, mode='constant')
            # # target = F.pad(target, p1d, mode='constant')
            # target_cpu = data[1][0, :H1, :W1, :T1].cpu().numpy() if scoring else None            
            # print('x_size',x.shape)            
            # logit = model(x) # nx5x9x9x9, target nx9x9x9
            # print('logit',logit.shape)        
            # output = F.softmax(logit, dim=1) # nx5x9x9x9

            # ## measure accuracy and record loss
            # #loss = None
            # #if scoring and criterion is not None:
            # #    loss = criterion(logit, target).item()

            # output = output[0, :, :H1, :W1, :T1].cpu().numpy()
        # else:
        print('x_size',x.shape)          
        _, logit = model(x) # nx5x9x9x9, target nx9x9x9
        print('logit',logit.shape)        
        output = F.softmax(logit, dim=1) # nx5x9x9x9

        ## measure accuracy and record loss
        #loss = None
        #if scoring and criterion is not None:
        #    loss = criterion(logit, target).item()

        output = output[0, :, :H, :W, :T].cpu().numpy()

        msg = 'Subject {}/{}, '.format(i+1, len(valid_loader))
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        if out_dir:
            np.save(os.path.join(out_dir, name + '_preds'), output)

        if scoring:
            output = output.argmax(0)
            scores = dice(output, target_cpu)

            #if loss is not None:
            #    scores += loss,

            vals.update(np.array(scores))

            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])

        if verbose:
            logging.info(msg)

    if scoring:
        msg = 'Average scores: '
        msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, vals.avg)])
        logging.info(msg)

    model.train()
    return vals.avg



def validate_ema(valid_loader, model,ckpt,ckpts_dir,
        out_dir='', names=None, scoring=True, verbose=True):
        
    model_file = os.path.join(ckpts_dir, ckpt)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['ema_state_dict'])
    H, W, T = 181, 217, 181   
    dtype = torch.float32

    dset = valid_loader.dataset

    model.eval()
    criterion = F.cross_entropy

    vals = AverageMeter()
    for i, data in enumerate(valid_loader):

        target_cpu = data[1][0, :H, :W, :T].numpy() if scoring else None
        data = [t.cuda(non_blocking=True) for t in data]

        x, target = data[:2]
        print('x_origin_size',x.shape)
        print('target',target.shape)
        if len(data) > 2:
            x = add_mask(x, data.pop(), 1)
        print('x_size',x.shape)
        # if(x.shape[4]==159):
            # print('pad to 160')
            # p1d = (0, 1)
            # x= F.pad(x, p1d, mode='constant')
            # # target = F.pad(target, p1d, mode='constant')
            # target_cpu = data[1][0, :H1, :W1, :T1].cpu().numpy() if scoring else None            
            # print('x_size',x.shape)            
            # logit = model(x) # nx5x9x9x9, target nx9x9x9
            # print('logit',logit.shape)        
            # output = F.softmax(logit, dim=1) # nx5x9x9x9

            # ## measure accuracy and record loss
            # #loss = None
            # #if scoring and criterion is not None:
            # #    loss = criterion(logit, target).item()

            # output = output[0, :, :H1, :W1, :T1].cpu().numpy()
        # else:
        print('x_size',x.shape)          
        _, logit = model(x) # nx5x9x9x9, target nx9x9x9
        print('logit',logit.shape)        
        output = F.softmax(logit, dim=1) # nx5x9x9x9

        ## measure accuracy and record loss
        #loss = None
        #if scoring and criterion is not None:
        #    loss = criterion(logit, target).item()

        output = output[0, :, :H, :W, :T].cpu().numpy()

        msg = 'Subject {}/{}, '.format(i+1, len(valid_loader))
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        if out_dir:
            np.save(os.path.join(out_dir, name + '_preds'), output)

        if scoring:
            output = output.argmax(0)
            scores = dice(output, target_cpu)

            #if loss is not None:
            #    scores += loss,

            vals.update(np.array(scores))

            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])

        if verbose:
            logging.info(msg)

    if scoring:
        msg = 'Average scores: '
        msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, vals.avg)])
        logging.info(msg)

    model.train()
    return vals.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', default='unet', type=str)
    parser.add_argument('-gpu', '--gpu', default='0', type=str)
    args = parser.parse_args()

    args = Parser(args.cfg, log='testing_550').add_args(args)

    #args.valid_list = 'valid_0.txt'
    #args.valid_list = 'all.txt'
    #args.saving = False
    #args.scoring = True

    args.data_dir = '/emc_lun/cgx/Script/U-Net_CNN/SemiSeg/code/BraTS2018/brats2018/testing'
    args.valid_list = 'test.txt'
    args.saving = True
    args.scoring = False # for real test data, set this to False

    # args.ckpt = 'model_epoch_550.tar'
    
    #args.ckpt = 'model_iter_227.tar'

    if args.saving:
        folder = os.path.splitext(args.valid_list)[0]
        out_dir = os.path.join('output', args.name, folder)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir
    else:
        args.out_dir = ''


    main()
