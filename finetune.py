import sys


import torch
import numpy as np
import random
from arguments import get_args
args = get_args()
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

from approaches import CSUR
from CSUR_model.model import BertClassifier_base
from dataset import bert_train_loader

from transformers import get_linear_schedule_with_warmup,AdamW
from transformers import BertForSequenceClassification
from torch.optim.lr_scheduler import OneCycleLR
epochs = 5
lr = 5e-5
batch_size = 16
regular = False



tasks = ['magazines.task','apparel.task','health_personal_care.task','camera_photo.task','toys_games.task','software.task','baby.task','kitchen_housewares.task','sports_outdoors.task','electronics.task','books.task','video.task',
    'imdb.task',
    'dvd.task',
    'music.task',
    'MR.task']
test_loader = []

for n,task in enumerate(tasks):
    model = BertClassifier_base()    
    train_dataloader,valid_dataloader,test_dataloader = bert_train_loader(tasks[n])
    test_loader.append(test_dataloader)
    model.add_dataset(tasks[n],2)
    model.set_dataset(dataset = tasks[n])
    model.cuda()

    optimizer = AdamW(filter(lambda p: p.requires_grad ,model.parameters()),
                lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                weight_decay= 1e-8
                )


    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    print(len(train_dataloader))
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    finetune_train = CSUR.Appr(model,epochs,batch_size,args = args,log_name=args.logname,task_names = tasks[n:n+1])
    print('train:'+tasks[n])
    finetune_train.train(0,train_dataloader,valid_dataloader,test_loader,optimizer,scheduler,regular)