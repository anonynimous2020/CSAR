import sys, time, os
import numpy as np
import random
import torch
from copy import deepcopy
import utils
from utils import *
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import *
import math

sys.path.append('..')
from arguments import get_args

args = get_args()

from bayes_layer import BayesianLinear,  _calculate_fan_in_and_fan_out
from transformers import get_linear_schedule_with_warmup,AdamW

import datetime
class Appr(object):
    

    def __init__(self, model, nepochs=100, sbatch=256, lr=0.001, 
                 lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100, args=None, log_name=None, split=False,task_names = None):

        self.model = model
        self.model_old = deepcopy(self.model)
        file_name = log_name
        self.logger = utils.logger(file_name=file_name, resume=False, path='result_data/csvdata/', data_format='csv')

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.clipgrad = clipgrad
        self.args = args
        self.iteration = 0
        self.split = split
        
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.tasks = task_names
        
        
        

        

        return



    def train(self, t, train_dataloader, test_dataloader, data,optimizer,scheduler,regular):
        best_loss = np.inf
        best_acc = 0
        best_model = utils.get_model(self.model)
        self.model_old = deepcopy(self.model)


        self.optimizer = optimizer
        self.scheduler = scheduler
        # initial best_avg
        valid_acc_t = {}
        valid_acc_t_norm = {}

        # Loop epochs
        for e in range(self.nepochs):
            

            # Train
            clock0 = time.time()
            avg = 0
            # num_batch = xtrain.size(0)
            num_batch = len(train_dataloader)
            self.model.set_dataset(self.tasks[t])
            self.train_epoch(t, train_dataloader,regular)
            
            clock1 = time.time()
            train_loss, train_acc = self.eval(t, train_dataloader,regular)
            
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / num_batch,
                1000 * self.sbatch * (clock2 - clock1) / num_batch, train_loss, 100 * train_acc), end='')
            # Valid
            
            valid_loss, valid_acc = self.eval(t, test_dataloader,regular)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

        
            if valid_acc >= best_acc:
                best_acc = valid_acc
                best_model = utils.get_model(self.model)
                # patience = self.lr_patience
                print(' *,best_model',end=" ")

            print()

            utils.freeze_model(self.model_old)  # Freeze the weights


        utils.set_model_(self.model, best_model)
        self.model_old = deepcopy(self.model)
        best_avg = 0
        for task in range(t+1):
            self.model.set_dataset(self.tasks[task])
            valid_loss_t, valid_acc_t[task] = self.eval(task, data[task],regular)
            best_avg += valid_acc_t[task]
            print('{} test: loss={:.3f}, acc={:5.1f}% |'.format(task,valid_loss_t, 100 * valid_acc_t[task]), end='')
            self.logger.add(epoch=(t * self.nepochs) + e+1, task_num=task + 1, test_loss=valid_loss_t,
                            test_acc=valid_acc_t[task])
        
        print('best_avg_Valid:  acc={:5.1f}% |'.format(100 * best_avg/(t+1)), end='')
        self.logger.add(task= t, avg_acc =100 * best_avg/(t+1) )
        self.logger.save()
        torch.save(self.model,'_task_{}.pt'.format(t))

        return
    def flat_accuracy(self,preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)



    def format_time(self,elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train_epoch(self,t,train_dataloader,regular):
        device = 'cuda'
        total_loss = 0
        self.model.train()
        for step, batch in enumerate(tqdm(train_dataloader,desc="train")):
        # for step,batch in enumerate(train_dataloader):
            

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            self.model.zero_grad()
            if regular:       
                outputs = self.model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels,
                            sample = True)
            else:
                outputs = self.model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]

            if t!=0 and regular:
                loss = self.custom_regularization(self.model_old, self.model, 16, loss)

            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)            
            

        return


    def eval(self,t,test_dataloader):

        t0 = time.time()
        device = 'cuda'

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():    
                if regular:    
                    outputs = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    sample = False)
                else:
                    outputs = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            loss = outputs[0]
            logits = outputs[1]
            eval_loss += loss.item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1
        # print(outputs[0].size())
        # Report the final accuracy for this validation run.
        avg_eval_loss = eval_loss/nb_eval_steps

        return avg_eval_loss, eval_accuracy/nb_eval_steps


# custom regularization

    def custom_regularization(self, saver_net, trainer_net, mini_batch_size, loss=None):
        
        sigma_weight_reg_sum = 0
        sigma_bias_reg_sum = 0
        sigma_weight_normal_reg_sum = 0
        sigma_bias_normal_reg_sum = 0
        mu_weight_reg_sum = 0
        mu_bias_reg_sum = 0
        L1_mu_weight_reg_sum = 0
        L1_mu_bias_reg_sum = 0
        
        out_features_max = 512
        
        
        prev_weight_strength = nn.Parameter(torch.Tensor(1,1).uniform_(0,0)).cuda()
        
        for (n, saver_layer), (_, trainer_layer) in zip(saver_net.named_modules(), trainer_net.named_modules()):
            
            if isinstance(trainer_layer, BayesianLinear)==False:
                continue
            # calculate mu regularization
            # print(n)
            trainer_weight_mu = trainer_layer.weight_mu
            saver_weight_mu = saver_layer.weight_mu
            trainer_bias = trainer_layer.bias
            saver_bias = saver_layer.bias
            
            fan_in, fan_out = _calculate_fan_in_and_fan_out(trainer_weight_mu)
            
            trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
            saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))
            
            if isinstance(trainer_layer, BayesianLinear):
                std_init = math.sqrt((2 / fan_in) * args.ratio)
            
            saver_weight_strength = (std_init / saver_weight_sigma)

            if len(saver_weight_mu.shape) == 4:
                out_features, in_features, _, _ = saver_weight_mu.shape
                curr_strength = saver_weight_strength.expand(out_features,in_features,1,1)
                prev_strength = prev_weight_strength.permute(1,0,2,3).expand(out_features,in_features,1,1)
            
            else:
                out_features, in_features = saver_weight_mu.shape
                curr_strength = saver_weight_strength.expand(out_features,in_features)
                if len(prev_weight_strength.shape) == 4:
                    feature_size = in_features // (prev_weight_strength.shape[0])
                    prev_weight_strength = prev_weight_strength.reshape(prev_weight_strength.shape[0],-1)
                    prev_weight_strength = prev_weight_strength.expand(prev_weight_strength.shape[0], feature_size)
                    prev_weight_strength = prev_weight_strength.reshape(-1,1)
                prev_strength = prev_weight_strength.permute(1,0).expand(out_features,in_features)
            
            L2_strength = torch.max(curr_strength, prev_strength)
            bias_strength = torch.squeeze(saver_weight_strength)
            
            L1_sigma = saver_weight_sigma
            bias_sigma = torch.squeeze(saver_weight_sigma)
            
            prev_weight_strength = saver_weight_strength
            
            mu_weight_reg = (L2_strength * (trainer_weight_mu-saver_weight_mu)).norm(2)**2
            mu_bias_reg = (bias_strength * (trainer_bias-saver_bias)).norm(2)**2
            
            L1_mu_weight_reg = (torch.div(saver_weight_mu**2,L1_sigma**2)*(trainer_weight_mu - saver_weight_mu)).norm(1)
            L1_mu_bias_reg = (torch.div(saver_bias**2,bias_sigma**2)*(trainer_bias - saver_bias)).norm(1)
            
            L1_mu_weight_reg = L1_mu_weight_reg * (std_init ** 2)
            L1_mu_bias_reg = L1_mu_bias_reg * (std_init ** 2)
            
            weight_sigma = (trainer_weight_sigma**2 / saver_weight_sigma**2)
            
            normal_weight_sigma = trainer_weight_sigma**2
            
            sigma_weight_reg_sum = sigma_weight_reg_sum + (weight_sigma - torch.log(weight_sigma)).sum()
            sigma_weight_normal_reg_sum = sigma_weight_normal_reg_sum + (normal_weight_sigma - torch.log(normal_weight_sigma)).sum()  
            
            mu_weight_reg_sum = mu_weight_reg_sum + mu_weight_reg
            mu_bias_reg_sum = mu_bias_reg_sum + mu_bias_reg
            L1_mu_weight_reg_sum = L1_mu_weight_reg_sum + L1_mu_weight_reg
            L1_mu_bias_reg_sum = L1_mu_bias_reg_sum + L1_mu_bias_reg
            
        # elbo loss
        loss = loss / mini_batch_size
        # L2 loss
        loss = loss + self.alpha * (mu_weight_reg_sum + mu_bias_reg_sum) / (2 * mini_batch_size)
        # L1 loss
        loss = loss + self.beta * (L1_mu_weight_reg_sum + L1_mu_bias_reg_sum) / (mini_batch_size)
        # sigma regularization
        loss = loss + self.gamma * (sigma_weight_reg_sum + sigma_weight_normal_reg_sum) / (2 * mini_batch_size)
        # print(self.beta)
            
        return loss

