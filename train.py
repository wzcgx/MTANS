import argparse
import os
import shutil
import time
import logging
import random
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader, sampler
import torch.optim as optim

from torch.autograd import Variable


import numpy as np
import sys
import models
from models import loss_fun
from data import datasets
from data.sampler import CycleSampler
from data.data_utils import add_mask, init_fn
from utils import Parser
from predict import validate, AverageMeter, validate_ema
from predict_all import validate as validate_all
from predict_all import validate_ema as validate_ema_all
from torch import nn
from utils import ramps
from models.discriminator import NetC
from utils.util import compute_sdf

parser = argparse.ArgumentParser()
#parser.add_argument('-cfg', '--cfg', default='unet_all', type=str)
#parser.add_argument('-cfg', '--cfg', default='unet_dice2_c25', type=str)
#parser.add_argument('-cfg', '--cfg', default='unet_dice2_redo', type=str)
parser.add_argument('-cfg', '--cfg', default='unet_ce_hard', type=str)
#parser.add_argument('-cfg', '--cfg', default='unet_bce_hard_per_im', type=str)
#parser.add_argument('-cfg', '--cfg', default='unet_bce_mean', type=str)
parser.add_argument('-gpu', '--gpu', default='1', type=str)
parser.add_argument('-out', '--out', default='', type=str)
parser.add_argument('-input_size', '--input_size', default='128,128,128', type=str,
					help="Comma-separated string with height and width of images.")
parser.add_argument('--beta', type=float,  default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--ema_decay', type=float,	default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,	 default=40.0, help='consistency_rampup')

# parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
					# help="Where restore model parameters from.")
# parser.add_argument("--restore-from-D", type=str, default=None,
					# help="Where restore model parameters from.")

					
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)
args.gpu = str(args.gpu)

ckpts = args.makedir()
resume = os.path.join(ckpts, 'model_epoch_500.tar')
resume_D = os.path.join(ckpts, 'model_D_epoch_500.tar')
#resume = ''
if not args.resume and os.path.exists(resume):
	args.resume = resume
	args.resume_D = resume_D


snapshot_path = "./model/" + args.cfg + "/"







def get_current_consistency_weight(epoch):
	# Consistency ramp-up from https://arxiv.org/abs/1610.02242
	return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
	# Use the true average until the exponential average is more correct
	alpha = min(1 - 1 / (global_step + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
		
		
def lr_poly(base_lr, iter, max_iter, power):
	return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
	lr = lr_poly(args.learning_rate, i_iter, args.max_iterations, args.power)
	optimizer.param_groups[0]['lr'] = lr
	if len(optimizer.param_groups) > 1 :
		optimizer.param_groups[1]['lr'] = lr * 10
		
def adjust_learning_rate_D(optimizer, i_iter):
	lr = lr_poly(args.learning_rate_D, i_iter, args.max_iterations, args.power)
	optimizer.param_groups[0]['lr'] = lr
	if len(optimizer.param_groups) > 1 :
		optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
	label = label.numpy()
	one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2], label.shape[3]), dtype=label.dtype)
	for i in range(args.num_classes):
		one_hot[:,i,...] = (label==i)
	#handle ignore labels
	return torch.FloatTensor(one_hot)
	
	
def make_D_label(label, ignore_mask):
	# ignore_mask = ignore_mask.data.cpu().numpy()
	ignore_mask = np.expand_dims(ignore_mask, axis=1)
	D_label = np.ones(ignore_mask.shape)*label
	D_label[ignore_mask] = 255
	D_label = Variable(torch.FloatTensor(D_label)).cuda()

	return D_label

def compute_argmax_map(output):
	output = output.detach().cpu().numpy()
	output1 = output
	print('output1 size' , np.shape(output1))
	output = output.transpose((1,2,3,0))
	output2 = output
	print('output2 size' , np.shape(output2))	 
	output = np.asarray(np.argmax(output, axis=3), dtype=np.int)
	output3 = output
	print('output3 size' , np.shape(output3))
	output = torch.from_numpy(output).float()
	output4 = output
	print('output4 size' , output4.size())	  
	return output
	 
def find_good_maps(D_outs, pred_all):
	count = 0
	for i in range(D_outs.size(0)):
		print('D_outs[i]:', D_outs[i])
		print('threshold_st', args.threshold_st)
		if D_outs[i] > args.threshold_st:
			count +=1

	if count > 0:
		print ('Above ST-Threshold : ', count, '/', args.batch_size)
		pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3), pred_all.size(4) )
		label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3), pred_sel.size(4))
		num_sel = 0 
		for j in range(D_outs.size(0)):
			if D_outs[j] > args.threshold_st:
				pred_sel[num_sel] = pred_all[j]
				print ('label_sel[num_sel] size: ',label_sel[num_sel].size())
				print ('compute_argmax_map(pred_all[j]) size: ',compute_argmax_map(pred_all[j]).size())
				label_sel[num_sel] = compute_argmax_map(pred_all[j])
				num_sel +=1
		return	pred_sel.cuda(), label_sel.cuda(), count  
	else:
		return 0, 0, count 
		
		
def inplace_relu(m):
	classname = m.__class__.__name__
	if classname.find('ReLU') != -1:
		m.inplace=True

# criterion = nn.BCELoss()



def main():

	h, w, z = map(int, args.input_size.split(','))
	input_size = (h, w, z)
	cudnn.enabled = True
	
	
	# make logger file
	if not os.path.exists(snapshot_path):
		os.makedirs(snapshot_path)
	# if os.path.exists(snapshot_path + '/code'):
		# shutil.rmtree(snapshot_path + '/code')
	# shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

	logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
						format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
	logging.info(str(args))




	# setup environments and seeds
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	# setup networks
	Network = getattr(models, args.net)
	# model = Network(**args.net_params)
	# model = model.cuda()

	
# # Load Plan  and Read Params
	# load_plans_file()	  

	
	
# Generic_Unet setting
	# model = Generic_UNet(num_input_channels, base_num_features, num_classes, net_numpool,
							# 2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
							# net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
							# net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)



	
	
	
	# model.train()
	# model = model.cuda()
	# model.inference_apply_nonlin = softmax_helper	   
	
	# cudnn.benchmark = True

	def create_model(ema=False):
		# Network definition
		Network = getattr(models, args.net)
		net = Network(**args.net_params)
		model = net.cuda()
		if ema:
			for param in model.parameters():
				param.detach_()
		return model

	model = create_model()

	ema_model = create_model(ema=True)



	# Network = getattr(models, args.net)
	# model = Network(**args.net_params)
	# model = model.cuda()
	
	
	# optimizer for segmentation network   


	optimizer = getattr(torch.optim, args.opt)(
		model.parameters(), **args.opt_params)
	optimizer.zero_grad()
	
	# optimizer = getattr(torch.optim, args.opt)(
		# model.parameters(), **args.opt_params)
	# optimizer.zero_grad()	   
	ce_loss = getattr(loss_fun, args.criterion)


	model_C = NetC(ngpu = 1)

	model_C.train()
	model_C.cuda()
	# model_C.apply(inplace_relu)	 
	# optimizer_D = optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
	optimizer_D = getattr(torch.optim, args.opt_D)(
		model_C.parameters(), **args.opt_params_D)	  


	# Dopt = optim.Adam(D.parameters(), lr=args.D_lr, betas=(0.9,0.99))	
	
	# init D

	# model_D = s4GAN_discriminator(num_classes=args.num_classes)


	# model_D.train()
	# model_D.cuda()


	# init D

	# model_D = s4GAN_discriminator(num_classes=args.num_classes)






	
	# optimizer = getattr(torch.optim, args.opt)(
			# model.parameters(), **args.opt_params)

	# lr_scheduler_eps = 1e-3
	# lr_scheduler_patience = 30
	# initial_lr = 3e-4
	# weight_decay = 3e-5
	# oversample_foreground_percent = 0.33 
	
	# optimizer = torch.optim.Adam(model.parameters(), initial_lr, weight_decay= weight_decay,
										  # amsgrad=True)
			
	# optimizer = getattr(torch.optim, args.opt)(
			# model.parameters(), **args.opt_params)			
	# criterion = getattr(criterions, args.criterion)

	msg = ''
	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_iter = checkpoint['iter']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optim_dict'])
			msg = ("=> loaded checkpoint '{}' (iter {})"
				  .format(args.resume, checkpoint['iter']))
		else:
			msg = "=> no checkpoint found at '{}'".format(args.resume)
	else:
		msg = '-------------- New training session ----------------'

	msg += '\n' + str(args)
	logging.info(msg)

	# Data loading code
	Dataset = getattr(datasets, args.dataset)
	
	
	all_train_list = os.path.join(args.data_dir, args.all_train_list)
	lab_train_list = os.path.join(args.data_dir, args.lab_train_list)
	unlab_train_list = os.path.join(args.data_dir, args.unlab_train_list)

	
	all_train_set = Dataset(all_train_list, root=args.data_dir, for_train=True,
			transforms=args.train_transforms)
	
	lab_train_set = Dataset(lab_train_list, root=args.data_dir, for_train=True,
			transforms=args.train_transforms)

	unlab_train_set = Dataset(unlab_train_list, root=args.data_dir, for_train=True,
			transforms=args.train_transforms)



				
	num_iters = args.num_iters or (len(all_train_set) * args.num_epochs) // args.batch_size
	num_iters -= args.start_iter
	train_sampler = CycleSampler(len(all_train_set), num_iters*args.batch_size)
	
	
	
	
	all_train_dataset_size = len(all_train_set)	   
	print ('train dataset size: ', all_train_dataset_size)
	   
	
	
	
	

	# train_dataset = train_set 

	lab_train_dataset = lab_train_set
	unlab_train_dataset = unlab_train_set



	lab_train_dataset_size = len(lab_train_set)
	unlab_train_dataset_size = len(unlab_train_set)
	# partial_size = int(args.labeled_ratio * train_dataset_size)

	# args.split_id = None
	# if args.split_id is not None:
		# train_ids = pickle.load(open(args.split_id, 'rb'))
		# print('loading train ids from {}'.format(args.split_id))
	# else:
		# train_ids = np.arange(train_dataset_size)
		# np.random.shuffle(train_ids)

	# pickle.dump(train_ids, open(os.path.join(ckpts, 'train_split.pkl'), 'wb'))

	# train_sampler = sampler.SubsetRandomSampler(train_ids[:partial_size])
	# pickle.dump(train_ids[:partial_size], open(os.path.join(ckpts, 'train_lab.pkl'), 'wb'))		 
	# train_remain_sampler = sampler.SubsetRandomSampler(train_ids[partial_size:])
	# pickle.dump(train_ids[partial_size:], open(os.path.join(ckpts, 'train_unlab.pkl'), 'wb'))			
	# train_gt_sampler = sampler.SubsetRandomSampler(train_ids[:partial_size])







	train_sampler = CycleSampler(len(lab_train_set), num_iters*args.batch_size)
	
	train_remain_sampler = CycleSampler(len(unlab_train_set), num_iters*args.batch_size)
	
	train_gt_sampler = CycleSampler(len(lab_train_set), num_iters*args.batch_size)

	# trainloader = DataLoader(lab_train_dataset,
					# batch_size=args.batch_size, sampler=lab_train_sampler, num_workers=4, pin_memory=True)
	# trainloader_remain = DataLoader(unlab_train_dataset,
					# batch_size=args.batch_size, sampler=unlab_train_sampler, num_workers=4, pin_memory=True)
	# trainloader_gt = DataLoader(lab_train_dataset,
					# batch_size=args.batch_size, sampler=lab_train_sampler, num_workers=4, pin_memory=True)
					
	# lab_train_ids = np.arange(lab_train_dataset_size)
	# np.random.shuffle(lab_train_ids)					  
					
	# unlab_train_ids = np.arange(unlab_train_dataset_size)
	# np.random.shuffle(unlab_train_ids)  

					
	# train_sampler = sampler.SubsetRandomSampler(lab_train_ids[:2])
	# pickle.dump(lab_train_ids[:2], open(os.path.join(ckpts, 'train_lab.pkl'), 'wb'))		  
	# train_remain_sampler = sampler.SubsetRandomSampler(unlab_train_ids[:13])
	# pickle.dump(unlab_train_ids[:13], open(os.path.join(ckpts, 'train_unlab.pkl'), 'wb'))			
	# train_gt_sampler = sampler.SubsetRandomSampler(lab_train_ids[:2])



	# trainloader = DataLoader(lab_train_dataset,
					# batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
	# trainloader_remain = DataLoader(unlab_train_dataset,
					# batch_size=args.batch_size, sampler=train_remain_sampler, num_workers=4, pin_memory=True)
	# trainloader_gt = DataLoader(lab_train_dataset,
					# batch_size=args.batch_size, sampler=train_gt_sampler, num_workers=4, pin_memory=True)


	trainloader = DataLoader(lab_train_dataset,
					batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
	trainloader_remain = DataLoader(unlab_train_dataset,
					batch_size=args.batch_size, sampler=train_remain_sampler, num_workers=4, pin_memory=True)
					
	# trainloader_remain = DataLoader(
		# unlab_train_dataset, batch_size=2, shuffle=False,
		# collate_fn=valid_set.collate,
		# num_workers=4, pin_memory=True)					 
					
	trainloader_gt = DataLoader(lab_train_dataset,
					batch_size=args.batch_size, sampler=train_gt_sampler, num_workers=4, pin_memory=True)



	trainloader_remain_iter = iter(trainloader_remain)

	trainloader_iter = iter(trainloader)
	trainloader_gt_iter = iter(trainloader_gt)



	if args.valid_list:
		valid_list = os.path.join(args.data_dir, args.valid_list)
		valid_set = Dataset(valid_list, root=args.data_dir,
				for_train=False, transforms=args.test_transforms)

		valid_loader = DataLoader(
			valid_set, batch_size=1, shuffle=False,
			collate_fn=valid_set.collate,
			num_workers=4, pin_memory=True)





	# optimizer for segmentation network
	# # optimizer = optim.SGD(model.parameters(),
				# # lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
	# # optimizer.zero_grad()

	# # optimizer for discriminator network
	# optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
	# optimizer_D.zero_grad()
	
	# loss/ bilinear upsampling	
	interp = nn.Upsample(size=(input_size[0], input_size[1], input_size[2]), mode='trilinear', align_corners=True)


	
	# labels for adversarial training
	pred_label = 0
	gt_label = 1

	y_real_, y_fake_ = Variable(torch.ones(args.batch_size, 1).cuda()), Variable(torch.zeros(args.batch_size, 1).cuda())
	Tanh = nn.Tanh()


	writer = SummaryWriter(snapshot_path+'/log')
	logging.info("{} itertations per epoch".format(len(trainloader)))
	
	
	start = time.time()

	enum_batches = len(all_train_set)/float(args.batch_size) #enum_batches = len(all_train_set)/float(args.batch_size) + 1 
	args.schedule	= {int(k*enum_batches): v for k, v in args.schedule.items()}
	args.save_freq	= int(enum_batches * args.save_freq)
	args.valid_freq = int(enum_batches * args.valid_freq)


	args.max_iterations = num_iters + args.start_iter
	print ('number of max iterations: ', args.max_iterations) 
	iter_num = 0
	max_epoch = args.max_iterations//len(trainloader)+1
	print ('number of max epoch: ', max_epoch)	   

	losses = AverageMeter()
	torch.set_grad_enabled(True)
	
	batch_dice = False
	# batch_dice = batch_dice
	initial_lr = args.learning_rate
	lr_ =  initial_lr
	
	# labels for adversarial training
	pred_label = 0
	gt_label = 1	
	max_par=0.0		   
	max_score_ema=0.0 
	max_score=0.0		 
	for i_iter in range(args.max_iterations):
	
		loss_ce_value = 0
		loss_D_value = 0
		loss_fm_value = 0
		loss_S_value = 0
		loss_G_value = 0
		loss_seg_value = 0
		loss_adv_pred_value = 0
		loss_D_value = 0
		loss_semi_value = 0
		loss_semi_adv_value = 0
		loss_consistency_value = 0
		ema_decay = 0.99
		consistency = 0.1
		consistency_rampup = 40.0
		loss_adv_value = 0
		loss_sdf_value = 0		  
		optimizer.zero_grad()
		adjust_learning_rate(optimizer, i_iter)
		optimizer_D.zero_grad()
		adjust_learning_rate_D(optimizer_D, i_iter)

		# #train C
		# model_C.zero_grad()

		# train with gt
		# get gt labels

		# training loss for labeled data only
		try:
			batch = next(trainloader_iter)
		except:
			trainloader_iter = iter(trainloader)
			batch = next(trainloader_iter)

		images, labels = batch
		images = Variable(images).cuda(non_blocking=True)
		labels = Variable(labels).cuda(non_blocking=True)
		images = torch.squeeze(images, 1)
		labels = torch.squeeze(labels, 1)
		# print ('images size: ', images.shape)
		# print ('labels dataset size: ', labels.shape)		   


		ignore_mask = (labels.cpu().numpy() == 255)
		ignore_mask_D = ignore_mask
		ignore_mask_gt = (labels.cpu().numpy() == 255)
		# pred = interp(model(images))
		# with torch.no_grad():
		
		# pred = model(images)	
		# pred_interp = interp(pred)		  




		# noise = torch.clamp(torch.randn_like(images) * 0.1, -0.2, 0.2)
		# ema_inputs = images + noise


		# with torch.no_grad():
			# ema_output = ema_model(ema_inputs)
		# ema_pred_interp = interp(ema_output)



		# ema_output = F.softmax(ema_output)
		# ema_output = ema_output.detach()

 
		# ema_pred_fore_1 = ema_output[:, 1]			   
  
 
		# flair = images
		# pred_gt = F.softmax(pred)
		# pred_gt = pred_gt.detach()
		# output_masked = flair.clone()
		# input_mask = flair.clone()




		# target_masked = torch.zeros([args.batch_size, 1, 128, 128, 128]).cuda()
		# target_masked[:, 0, :] = input_mask[:,0,:,:,:] * ema_pred_fore_1

		
		# target_masked = target_masked.cuda()
		# target_masked_lab = target_masked 






		
		# pred_fore_1_gt = pred_gt[:, 1]

		
		
		# labels_n_gt = labels.cpu().numpy()	
		# labels_n_gt_1 = labels_n_gt
	  
		# labels_n_gt_1[(labels_n_gt_1==2)]=0
		# labels_n_gt_1[(labels_n_gt_1==3)]=0			   
		# labels_fore_gt_1 = torch.tensor(labels_n_gt_1)
		# labels_fore_gt_1 = labels_fore_gt_1.cuda().float()

  
		
		# #detach G from the network

		# output_masked = torch.zeros([args.batch_size, 1, 128, 128, 128]).cuda()
		# output_masked[:, 0, :] = input_mask[:,0,:,:,:] * pred_fore_1_gt

		# output_masked = output_masked.cuda()
		# # print('output_masked:', output_masked.shape)
		# output_masked_lab = output_masked
		# target_masked = flair.clone()

		# target_masked = torch.zeros([args.batch_size, 1, 128, 128, 128]).cuda()

		# target_masked[:, 0, :] = input_mask[:,0,:,:,:] * labels_fore_gt_1

		# target_masked = target_masked.cuda()

		# pred_gt_cat = torch.cat((F.softmax(pred_interp,dim=1), output_masked), dim=1)
		# _, result_gt = model_C(pred_gt_cat)

		# D_gt_v_gt = Variable(one_hot(labels)).cuda(non_blocking=True)
		# # print('D_gt_v_gt',D_gt_v_gt.shape)
		# D_gt_v_cat_gt = torch.cat((D_gt_v_gt, target_masked), dim=1)
		

		# _, target_D = model_C(D_gt_v_cat_gt)		
		# loss_DD = - torch.mean(torch.abs(result_gt - target_D))
		# loss_DD.backward()
		# optimizer_D.step()

		
		# #clip parameters in D
		# for p in model_C.parameters():
			# p.data.clamp_(-0.05, 0.05)
		

		# model.zero_grad()	


		outputs_tanh_1, pred = model(images)

		with torch.no_grad():
			gt_dis = compute_sdf(labels.cpu().numpy(), pred.shape)
			gt_dis = torch.from_numpy(gt_dis).float().cuda()
		loss_sdf = F.mse_loss(outputs_tanh_1, gt_dis)


		
		loss_seg = ce_loss(pred, labels)

		loss = loss_seg

		losses.update(loss.item(), labels.numel())

		pred_gt = F.softmax(pred, dim=1)



		output_masked = images.clone()
		input_mask = images.clone()
		pred_fore_1_gt = pred_gt[:, 1]
		# pred_fore_2_gt = pred_gt[:, 2]  
		# pred_fore_3_gt = pred_gt[:, 3] 
		
		
		labels_n_gt = labels.cpu().numpy()	
		labels_n_gt_1 = labels_n_gt
		labels_n_gt_2 = labels_n_gt
		labels_n_gt_3 = labels_n_gt		   
		labels_n_gt_1[(labels_n_gt_1==2)]=0
		labels_n_gt_1[(labels_n_gt_1==3)]=0		   
		# labels_n_gt_2[(labels_n_gt_2==1)]=0
		# labels_n_gt_2[(labels_n_gt_2==3)]=0
		# labels_n_gt_2[(labels_n_gt_2==2)]=1		 
		# labels_n_gt_3[(labels_n_gt_3==1)]=0
		# labels_n_gt_3[(labels_n_gt_3==2)]=0
		# labels_n_gt_3[(labels_n_gt_3==3)]=1		 
		labels_fore_gt_1 = torch.tensor(labels_n_gt_1)
		labels_fore_gt_1 = labels_fore_gt_1.cuda().float()
		# labels_fore_gt_2 = torch.tensor(labels_n_gt_2)
		# labels_fore_gt_2 = labels_fore_gt_2.cuda().float()
		# labels_fore_gt_3 = torch.tensor(labels_n_gt_3)
		# labels_fore_gt_3 = labels_fore_gt_3.cuda().float() 
		
		
		output_masked = torch.zeros([args.batch_size, 1, 128, 128, 128]).cuda()
		output_masked[:, 0, :] = input_mask[:,0,:,:,:] * pred_fore_1_gt
		# output_masked[:, 1, :] = input_mask[:,0,:,:,:] * pred_fore_2_gt
		# output_masked[:, 2, :] = input_mask[:,0,:,:,:] * pred_fore_3_gt
		
		output_masked = output_masked.cuda()
		output_masked_lab = output_masked
		# print('outputs_tanh_1',outputs_tanh_1)
		result,_ = model_C(outputs_tanh_1, output_masked)
		target_masked = images.clone()

		target_masked = torch.zeros([args.batch_size, 1, 128, 128, 128]).cuda()
		target_masked[:, 0, :] = input_mask[:,0,:,:,:] * labels_fore_gt_1
		# target_masked[:, 1, :] = input_mask[:,0,:,:,:] * labels_fore_gt_2
		# target_masked[:, 2, :] = input_mask[:,0,:,:,:] * labels_fore_gt_3

		target_masked = target_masked.cuda()
		# print('labels',labels.shape)

	   
		# print('labels_tanh',labels_tanh)
		labels_tanh = Variable(one_hot(labels.cpu())).cuda(non_blocking=True)		   

		target_G,_ = model_C(labels_tanh, target_masked)		 
		
		loss_G = torch.mean(torch.abs(result - target_G))		 
		










		# print('trainloader_remain_iter', trainloader_remain_iter)		   
		try:
			batch_remain = next(trainloader_remain_iter)
		except:
			trainloader_remain_iter = iter(trainloader_remain)
			batch_remain = next(trainloader_remain_iter)
		
		images_remain, _ = batch_remain
		# print ('images remain size: ', images_remain.shape)			 
		images_remain = Variable(images_remain).cuda(non_blocking=True)
		images_remain = torch.squeeze(images_remain, 1)


		# Generate Discriminator target based on sampler
		Dtarget = torch.tensor([1, 0]).cuda()
		model.train()
		model_C.eval()


# noise and ema_output
		noise = torch.clamp(torch.randn_like(images_remain) * 0.1, -0.2, 0.2)
		ema_inputs = images_remain + noise


		with torch.no_grad():
			outputs_tanh, pred_output = model(images_remain)

			
		with torch.no_grad():
			ema_outputs_tanh, ema_output = ema_model(ema_inputs)


		pred_output = F.softmax(pred_output)


		ema_output = F.softmax(ema_output)


		pred_fore_1 = pred_output[:, 1]	 
		ema_pred_fore_1 = ema_output[:, 1]			   
  
		# pred_fore_2 = pred_output[:, 2]  
		# ema_pred_fore_2 = ema_output[:, 2]	   
  
		# pred_fore_3 = pred_output[:, 3]  
		# ema_pred_fore_3 = ema_output[:, 3]  

# # uncertainty
		# T = 8
		# volume_batch_r = images_remain.repeat(2, 1, 1, 1, 1)
		# stride = volume_batch_r.shape[0] // 2
		# preds = torch.zeros([stride * T, 4, 128, 128, 128]).cuda()
		# for i in range(T//2):
			# ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
			# with torch.no_grad():
				# ema_out = ema_model(ema_inputs)
				# preds[2 * stride * i:2 * stride * (i + 1)] = ema_out

		# preds = F.softmax(preds, dim=1)
		# preds = preds.reshape(T, stride, 4, 128, 128,128)
		# preds = torch.mean(preds, dim=0)	#(batch, 2, 112,112,80)
		# uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) 





# calculate the loss
		epo = int((i_iter+1) // enum_batches)
		consistency_weight = get_current_consistency_weight(epo)
		adv_weight = get_current_consistency_weight(i_iter//150)
		# consistency_dist = softmax_mse_WT_loss(pred_output, ema_output) #(batch, 2, 112,112,80)
		
		image_flair = images_remain[:, 0:1]
		output_masked = image_flair.clone()
		input_mask = image_flair.clone()
		



  

		output_masked = torch.zeros([args.batch_size, 1, 128, 128, 128]).cuda()
		output_masked[:, 0, :] = input_mask[:,0,:,:,:] * pred_fore_1
		# output_masked[:, 1, :] = input_mask[:,0,:,:,:] * pred_fore_2
		# output_masked[:, 2, :] = input_mask[:,0,:,:,:] * pred_fore_3

		
		output_masked = output_masked.cuda()
		output_masked_unlab = output_masked

		# Doutputs = D(outputs_tanh, images_remain)
		
		

		
		result_remain, Doutputs = model_C(outputs_tanh, output_masked)
		target_masked = image_flair.clone()

		target_masked = torch.zeros([args.batch_size, 1, 128, 128, 128]).cuda()
		target_masked[:, 0, :] = input_mask[:,0,:,:,:] * ema_pred_fore_1
		# target_masked[:, 1, :] = input_mask[:,0,:,:,:] * ema_pred_fore_2
		# target_masked[:, 2, :] = input_mask[:,0,:,:,:] * ema_pred_fore_3
		
		target_masked = target_masked.cuda()
		
		
		
		target_C, ema_Doutputs = model_C(ema_outputs_tanh, target_masked)
		
		
		
		consistency_dist = torch.mean(torch.abs(result_remain - target_C))


		
		
		
		# threshold = (0.75+0.25*ramps.sigmoid_rampup(i_iter, args.max_iterations))*np.log(2)
		# mask = (uncertainty<threshold).float()
		# consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
		consistency_loss = consistency_weight * consistency_dist
		



		print('consistency_loss', consistency_loss.requires_grad)

		print('loss_G', loss_G.requires_grad)
		print('loss_seg', loss_seg.requires_grad)


		loss_adv = F.cross_entropy(Doutputs, Dtarget[:1].long())

		loss = loss_seg	 + loss_G + consistency_loss + args.beta * loss_sdf +  adv_weight * loss_adv
		

		
		
		loss.backward()
		
		loss_seg_value += loss_seg.data.cpu().numpy() 
		loss_G_value += loss_G.data.cpu().numpy()
		loss_consistency_value += consistency_loss.cpu().detach().numpy()
		
		loss_adv_value += loss_adv.cpu().detach().numpy()
		loss_sdf_value += loss_sdf.cpu().detach().numpy()		



	   
		optimizer.step()				   
		update_ema_variables(model, ema_model, ema_decay, i_iter)		 


		# Train D
		model.eval()
		model_C.train()

		with torch.no_grad():						 
			outputs_tanh_3, outputs_3 = model(images)
			outputs_tanh_4, outputs_4 = model(images_remain)
		output_masked_lab = output_masked_lab.detach()
		output_masked_unlab = output_masked_unlab.detach()	 
		outputs_tanh = torch.cat((outputs_tanh_3, outputs_tanh_4), 0)
		volume_batch  = torch.cat((output_masked_lab, output_masked_unlab), 0)
		result_D, Doutputs = model_C(outputs_tanh, volume_batch)
		# D want to classify unlabel data and label data rightly.
		# print('Doutputs',Doutputs.shape)		  
		loss_D_2 = F.cross_entropy(Doutputs, Dtarget.long())



		# pred_lab = outputs_3
		# pred_lab_soft = F.softmax(pred_lab, dim=1)
		input_mask = images.clone()
		labels_gt = labels.clone()	
		# pred_fore_1_gt = pred_lab_soft[:, 1,:,:,:]
		labels_n_gt_1 = labels_gt.cpu().numpy()
		labels_n_gt_1[(labels_n_gt_1==2)]=0
		labels_n_gt_1[(labels_n_gt_1==3)]=0	
		labels_fore_gt_1 = torch.tensor(labels_n_gt_1)
		labels_fore_gt_1 = labels_fore_gt_1.cuda().float()
		
		# output_masked = input_mask * pred_fore_1_gt
		# output_masked_lab = output_masked

		target_masked = torch.zeros([args.batch_size, 1, 128, 128, 128]).cuda()
		target_masked[:, 0, :] = input_mask[:,0,:,:,:] * labels_fore_gt_1
		target_masked = target_masked.cuda()
		
		# output_masked_lab = output_masked
		target_masked_lab = target_masked

		D_gt_v_gt = Variable(one_hot(labels_gt.cpu())).cuda(non_blocking=True)
		# D_gt_v_cat_gt = torch.cat((D_gt_v_gt, target_masked), dim=1)					
		

		# pred_interp = interp(pred_lab)
		# pred_remain = interp(pred_unlab)			 
		# pred_interp = torch.cat((pred_interp, pred_remain), 0)
		# output_masked_2 = torch.cat((output_masked_lab, output_masked_unlab), 0)			
		# ignore_mask = np.concatenate((ignore_mask_lab,ignore_mask_remain), axis = 0)
		# D_gt_v_cat = torch.cat((F.softmax(pred_interp, dim=1), output_msaked_2), dim=1)	

		# result_gt, _ = model_C(outputs_tanh_5, output_masked_2)


		target_D, _ = model_C(D_gt_v_gt, target_masked)
			
		loss_D_dis = - torch.mean(torch.abs(result_D[:1] - target_D))


		loss_D = loss_D_2 + loss_D_dis


		# Dtp and Dfn is unreliable because of the num of samples is small(4)
		# Dacc = torch.mean((torch.argmax(Doutputs, dim=1).float()==Dtarget.float()).float())
		# Dtp = torch.mean((torch.argmax(Doutputs, dim=1).float()==Dtarget.float()).float())
		# Dfn = torch.mean((torch.argmax(Doutputs, dim=1).float()==Dtarget.float()).float())
		# Dopt.zero_grad()
		
		
		optimizer_D.zero_grad()
		loss_D.backward()
		optimizer_D.step()







		lr_S = optimizer.param_groups[0]['lr']
		lr_D = optimizer_D.param_groups[0]['lr']

		writer.add_scalar('lr/lr_S', lr_S, i_iter)
		writer.add_scalar('loss/loss_seg', losses.avg, i_iter)


		writer.add_scalar('loss/consistency_loss', consistency_loss, i_iter)
		writer.add_scalar('loss/loss_sdf', loss_sdf, i_iter)
		writer.add_scalar('loss/loss_adv', loss_adv, i_iter) 
		writer.add_scalar('loss/loss_G', loss_adv, i_iter)        
		writer.add_scalar('train/consistency_weight', consistency_weight, i_iter)
		writer.add_scalar('train/consistency_dist', consistency_dist, i_iter)		 


		# print('iter = {0:8d}/{1:8d}, loss_dc_ce = {2:.3f}, loss_fm = {3:.3f}, loss_S = {4:.3f}, loss_D = {5:.3f}'.format(i_iter, args.max_iterations, loss_ce_value, loss_fm_value, loss_S_value, loss_D_value))
		
		
		msg = 'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_G = {3:.3f}, consistency_loss = {4:.6f}, consistency_weight = {5:.6f}, consistency_dist = {6:.6f}, loss_sdf = {7:.6f}, loss_adv = {8:.6f}, adv_weight = {9:.6f}'.format(i_iter, args.max_iterations, losses.avg, loss_G_value, loss_consistency_value, consistency_weight, consistency_dist, loss_sdf_value, loss_adv_value, adv_weight)
		
		# msg = 'iter = {0:8d}/{1:8d}, Loss {2:.4f}'.format(
				 # i_iter, args.max_iterations, losses.avg)
		
		
		
		
		
		# msg = 'iter = {0:8d}/{1:8d}, loss_dc_ce = {2:.3f}, loss_fm = {3:.3f}, loss_S = {4:.3f}, loss_D = {5:.3f}'.format(i_iter, args.max_iterations, loss_ce_value, loss_fm_value, loss_S_value, loss_D_value)
		logging.info(msg)		 

		if (i_iter+1) % args.save_freq == 0:
			epoch = int((i_iter+1) // enum_batches)
			file_name = os.path.join(ckpts, 'model_epoch_{}.tar'.format(epoch))
			file_name_D = os.path.join(ckpts, 'model_D_epoch_{}.tar'.format(epoch))			   
			torch.save({
				'iter': i_iter+1,
				'state_dict': model.state_dict(),
				'ema_state_dict': ema_model.state_dict(),								 
				'optim_dict': optimizer.state_dict(),
				},
				file_name)
		  



		

	i = num_iters + args.start_iter
	file_name = os.path.join(ckpts, 'model_last.tar')
	file_name_D = os.path.join(ckpts, 'model_D_last.tar')	 
	torch.save({
		'iter': i,
		'state_dict': model.state_dict(),
		'ema_state_dict': ema_model.state_dict(),						 
		'optim_dict': optimizer.state_dict(),
		},
		file_name)
	   

	if args.valid_list:
		logging.info('-'*50)
		# msg  =  'Best_Epoch {:.4f}, {}'.format(best_epoch, 'validate validation data')
		logging.info(msg)

		msg	 =	'{}'.format('validate validation data with the model')
		logging.info(msg)
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(50)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 50			   
				max_score = dice_avg_score
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(100)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 100			  
				max_score = dice_avg_score
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(150)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 150			  
				max_score = dice_avg_score
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(200)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 200			  
				max_score = dice_avg_score			  
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(250)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 250			  
				max_score = dice_avg_score
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(300)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 300			  
				max_score = dice_avg_score
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(350)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 350			  
				max_score = dice_avg_score

		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(400)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 400			  
				max_score = dice_avg_score			  
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(450)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 450			  
				max_score = dice_avg_score			 
		
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(500)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 500			  
				max_score = dice_avg_score			  
			
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(550)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 550			  
				max_score = dice_avg_score			 
			
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(600)
			ckpts_dir = args.getdir()
			dice_avg_score = validate(valid_loader, model, ckpt,ckpts_dir,	names=valid_set.names, out_dir=args.out)
			if (dice_avg_score>=max_score):
				best_epoch = 600			  
				max_score = dice_avg_score
		msg = 'MAX-------------------'
		logging.info(msg)	 
		# print('best_epoch',best_epoch)
		# print('max_score',max_score)
		# msg = 'Epoch_MAX = {0:8d}, Score_Max = {1:.4f}'.format(best_epoch, max_score)
		msg = 'Epoch_MAX = {0}, WT_MAX = {1}'.format(best_epoch, max_score)		   
		logging.info(msg)

		msg	 =	'{}'.format('validate validation data with the EMA model')
		logging.info(msg)
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(50)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 50
				max_score_ema = dice_avg
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(100)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 100
				max_score_ema = dice_avg
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(150)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 150			
				max_score_ema = dice_avg
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(200)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 200			
				max_score_ema = dice_avg			
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(250)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 250			
				max_score_ema = dice_avg
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(300)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 300			
				max_score_ema = dice_avg
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(350)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 350			
				max_score_ema = dice_avg

		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(400)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 400			
				max_score_ema = dice_avg			
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(450)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 450			
				max_score_ema = dice_avg		   
		
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(500)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 500			
				max_score_ema = dice_avg		   
			
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(550)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 550			
				max_score_ema = dice_avg			
			
		with torch.no_grad():
			ckpt = 'model_epoch_{}.tar'.format(600)
			ckpts_dir = args.getdir()
			dice_avg = validate_ema(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out)
			if (dice_avg>=max_score_ema):
				best_epoch_ema = 600			
				max_score_ema = dice_avg

	msg = 'MAX------------------------ '
	logging.info(msg)		 
	msg = 'Epoch_MAX_ema = {0}, Score_Max_ema = {1}'.format(best_epoch_ema, max_score_ema)
	logging.info(msg)	 


	if (max_score>=max_score_ema):

			best_epoch_all = best_epoch
			best_score_all = max_score
			ema_true = 0
	else:
			best_epoch_all = best_epoch_ema
			best_score_all = max_score_ema
			ema_true = 1

	msg = 'Best_Epoch_ALL= {0}, Best_Score_ALL= {1}'.format(best_epoch_all, best_score_all)
	logging.info(msg) 
		  
	msg = 'total time: {:.4f} minutes'.format((time.time() - start)/60)
	logging.info(msg)


	if ema_true==0:
		if args.valid_list:
			logging.info('-'*50)
			logging.info(msg)
			msg	 =	'{}'.format('validate validation data with the model')
			logging.info(msg)			 
			with torch.no_grad():
				ckpt = 'model_epoch_{}.tar'.format(best_epoch_all)
				ckpts_dir = args.getdir()
				validate_all(valid_loader, model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out_dir)			   
			

		msg = 'total time: {:.4f} minutes'.format((time.time() - start)/60)
		logging.info(msg)
	if ema_true==1:
		if args.valid_list:
			logging.info('-'*50)
			logging.info(msg)
			msg	 =	'{}'.format('validate validation data with the ema model')
			logging.info(msg)			 
			with torch.no_grad():
				ckpt = 'model_epoch_{}.tar'.format(best_epoch_all)
				ckpts_dir = args.getdir()
				validate_ema_all(valid_loader, ema_model, ckpt,ckpts_dir,  names=valid_set.names, out_dir=args.out_dir)	 

if __name__ == '__main__':
	main()
