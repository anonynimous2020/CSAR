import sys
# sys.path.append('CSAR/')

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

from approaches import CSAR
from CSAR_model.model import BertClassifier_PALS,BertClassifier_CSAR,BertClassifier_base,BertClassifier_UCL,BertClassifier_CSAR_frezzbert,LSTMSentiment,CNN_Text
from dataset import bert_train_loader

from transformers import get_linear_schedule_with_warmup,AdamW
from transformers import BertForSequenceClassification
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn



epochs = 3
lr = 5e-5
batch_size = 16
regular = False
if args.approach == 'BERT':
    model = BertClassifier_base()
elif args.approach == 'CSAR':
    model = BertClassifier_CSAR()
    regular = True
elif args.approach == 'PALS':
    model = BertClassifier_PALS()
elif args.approach == 'UCL': 
    model = BertClassifier_UCL()
    regular = True
elif args.approach == 'FreezeBert':
    model = BertClassifier_CSAR_frezzbert()
elif args.approach == 'LSTM':
    model = LSTMSentiment(768,256,30522,True,16)
    lr = 1e-3
    epochs = 5
    batch_size = 16
elif args.approach == 'CNN_Text':
    model = CNN_Text()
    lr = 1e-3
    epochs = 5
    batch_size = 16

print("trainning :"+args.approach)

model.cuda()
# model = nn.DataParallel(model , device_ids=[0,1])
# model = model.module

test_loader=[]

tasks = ['magazines.task','apparel.task','health_personal_care.task','camera_photo.task','toys_games.task','software.task','baby.task','kitchen_housewares.task','sports_outdoors.task',
    'electronics.task','books.task','video.task','imdb.task','dvd.task','music.task','MR.task']
if args.tasks_sequence ==2:
    tasks = ['health_personal_care.task', 'books.task', 'magazines.task', 'music.task', 'baby.task', 'software.task', 'camera_photo.task', 'sports_outdoors.task', 'kitchen_housewares.task', 'video.task', 'MR.task', 'apparel.task', 'imdb.task', 'dvd.task', 'electronics.task', 'toys_games.task']
print(tasks)
CSAR_train = CSAR.Appr(model,epochs,batch_size,args = args,log_name=args.logname,task_names = tasks)


for t in range(0,16):
    train_dataloader,valid_dataloader,test_dataloader = bert_train_loader(tasks[t])
    test_loader.append(test_dataloader)
    model.add_dataset(tasks[t],2)
    model.set_dataset(dataset = tasks[t])

    model.cuda()

    pals_p_id = []
    for n,param in model.named_parameters():
        if 'mult' in n:
            pals_p_id.append(id(param))
    bert_param = filter(lambda p: p.requires_grad and id(p) not in pals_p_id ,model.parameters())
    pals_param = filter(lambda p: p.requires_grad and id(p) in pals_p_id ,model.parameters())
    params = [
        {"params":bert_param,"lr":lr},
        {"params":pals_param,"lr":1e-3},        
    ]
    optimizer = AdamW(params,
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


    print('training {}'.format(tasks[t]))
    CSAR_train.train(t,train_dataloader,valid_dataloader,test_loader,optimizer,scheduler,regular)
