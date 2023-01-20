import datetime
import os.path as osp
import torch
import numpy as np
import tqdm
import math
import pytz
import os
import imageio
import shutil
from importlib import import_module
import matplotlib.pyplot as plt
from models.pointnet2 import Pointnet2
from models.pointnet2_cls_msg_ppf import get_model





class Trainer():
    def __init__(self, data_loader, opts, b_size, vis = None):
        self.cuda = opts.cuda
        self.opts = opts
        self.train_loader = data_loader[0]
        self.val_loader = data_loader[1]
        self.scheduler = []
        self.b_size = b_size
        self.log_dir = opts.log_dir
        
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = get_model(40)
        self.loss_func = torch.nn.NLLLoss()        

        # model_module = import_module('models.{}.fcn{}'.format(
        #     opts.backbone, opts.fcn))
        # self.model = model_module.FCN(n_class=1)


        if opts.mode == 'train':
            if opts.optimizer == 'Adam':
                self.optim = torch.optim.Adam(
                    self.model.parameters(),
                    lr=opts.cfg['lr'],
                    betas=opts.cfg['betas'],
                    eps=1e-8,
                    weight_decay = opts.cfg['weight_decay']
                )
            else:
                self.optim = torch.optim.SGD(self.model.parameters(), lr=opts.cfg['lr'], momentum=0.9)
        load_ckpt = opts.resume.lower() in ("yes", "true", "t", "1", 'True', 'TRUE')
        if load_ckpt:
            print('loaded ckpt: ', self.log_dir+"/model_best.pth.tar")
            state_dict = torch.load(self.log_dir+"/model_best.pth.tar")
            self.model.load_state_dict(state_dict)


        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.epoch = 0
        #self.model.to(self.device)
        self.iteration = 0
        self.iter_val = 0
        self.max_iter = opts.cfg['max_iteration']
        self.best_acc_mean = 999999999999999999
        #visualizer
        self.vis = vis

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Canada/Central'))

        self.interval_validate = opts.cfg.get('interval_validate',
                                              len(self.train_loader))
        if self.interval_validate is None:
            self.interval_validate = len(self.train_loader)

        self.out = opts.out
        if not osp.exists(self.out):
            os.makedirs(self.out)

    def validate(self):
        # import matplotlib.pyplot as plt
        training = self.model.training
        self.model.eval()
        mean_acc_sum = 0
        mean_center_off_sum = 0
        #iteration = 0
        visualizations = []
        label_trues, label_preds = [], []
        internal_iter=0
        correct=0
        total=0
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm.tqdm(
                    enumerate(self.val_loader),
                    total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration,
                    ncols=80,
                    leave=False):
                    
                data, target= data.to(self.device), target.to(self.device)
                if(data.shape[0]==1): continue 
                score = self.model(data)
                loss = self.loss_func(score, target.long())
                #print(loss)
                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while validating')
                else:
                    mean_acc_sum += float(loss.item())
                self.iter_val += 1
                internal_iter += 1
                scores_np = score.cpu().detach().numpy()
                target_np = target.cpu().detach().numpy()
                for i in range(scores_np.shape[0]):
                    idx=np.where(scores_np[i]==scores_np[i].max())
                    if (len(idx)==1):
                        if (target_np[i]==idx[0][0]):
                            correct+=1
                    total+=1

                #print(correct,total)

                if self.vis is not None:
                    self.vis.add_scalar('Val', float(loss.item()), self.iter_val)
                    
        mean_acc = mean_acc_sum / float(internal_iter)
        self.vis.add_scalar('Accuracy', correct/total,self.epoch)
        self.scheduler.step(loss)
        #print('validation mean acc:',mean_acc)
        is_best = mean_acc < self.best_acc_mean
        if is_best:
            self.best_acc_mean = mean_acc
        #save_name = str(datetime.datetime.now(pytz.timezone('Canada/Central')).strftime('%Y-%m-%d_%H:%M:%S'))+"__ckpt.pth.tar"
        save_name = "ckpt.pth.tar"
        if torch.cuda.device_count() > 1:
          torch.save(self.model.module.state_dict(), osp.join(self.out, save_name))
        else:
          torch.save(self.model.state_dict(), osp.join(self.out, save_name))
        if is_best:
            shutil.copy(osp.join(self.out, save_name),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch,
                ncols=80,
                leave=False):
            data, target= data.to(self.device), target.to(self.device)
            #print(data.shape)
            #print(target.shape)
            if(data.shape[0]==1): 
                print('batch size = 1, ignored')
                continue # ignore when batch size=1 to resolve the bn issue
            #print(data.shape)
            self.optim.zero_grad()
            score = self.model(data)
            score = score.to(self.device)
            #print(score.shape)
            #print(target)
            
            loss = self.loss_func(score, target.long())
            loss.backward()
            self.optim.step()

            np_loss = loss.detach().cpu().numpy()
            #print(np_loss)
            if np.isnan(np_loss):
                raise ValueError('loss is nan while training')

            #visulalization
            if self.vis is not None:
                self.vis.add_scalar('Train', np_loss, self.iteration)
            self.iteration+=1

            if self.iteration >= self.max_iter:
                break

    def Train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, verbose=True)
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train',
                                 ncols=80):
            self.epoch = epoch
            
            self.train_epoch()
            #if epoch % 5 == 0:
            self.validate()
            
            if self.iteration >= self.max_iter:
                break

