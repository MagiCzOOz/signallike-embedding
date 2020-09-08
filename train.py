#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:33:35 2020

@author: prang

Usage:
    main.py [-h | --help]
    main.py [--version]
    main.py [--gpu] [--gpudev GPUDEVICE] [--lr LR] [--maxiter MITER]
            [--runname RNAME] [--inputrep REP] [--path P] [--bsize BSIZE]
            [--nbframe NBFRAME] [--o OUT] [--save]
    
Options:
    -h --help  Show this helper
    --version  Show version and exit
    --gpu  Use GPU or not [default: False]
    --gpudev GPUDEVICE  Which GPU will be use [default: 0]
    --lr LR  Initial learning rate [default: 1e-4]
    --maxiter MITER  Maximum number of updates [default: 50]
    --runname RNAME  Set the name of the run for tensorboard [default: default_run]
    --inputrep REP  Set the representation which will be used as input [default: midilike]
    --path P  The path of the MIDI files folder (with a test and train folder) \
            [default: /fast-1/mathieu/datasets/Chorales_Bach_Proper_with_all_transposition].
    --bsize BSIZE  Batch size [default: 16]
    --nbframe NBFRAME  Number of frames per bar [default: 16]
    --o OUT  Path of the output directory [default: None]
    --save  Save the models during the training or not [default: True]
"""

import torch
import representations as rep_classes
import models as m
from tensorboardX import SummaryWriter
from docopt import docopt
from tqdm import tqdm
import os


# Usefull functions
def increase_wkl(epoch, w_kl, input_rep):
    
    if input_rep == "pianoroll":
        if epoch < 150  and epoch > 0:
            if epoch % 10 == 0:
                w_kl += 1e-5
        else :
            if epoch % 10 == 0 :
                w_kl += 1e-4
    elif input_rep in ["midilike", "signallike"]:
        if epoch % 10 == 0 and epoch > 0:
            w_kl += 1e-8
    elif input_rep == "midimono":
        if epoch % 10 == 0 and epoch > 0:
            w_kl += 1e-4
    elif input_rep == "notetuple":
        if epoch % 10 == 0 and epoch > 0:
            w_kl += 1e-6
    
    return w_kl


# Load arguments from the shell command with docopt
if __name__ == '__main__':
    arguments = docopt(__doc__, version='symbolic_embeddings v1.0')
    print(arguments)    
    if torch.cuda.is_available() and not arguments['--gpu']:
            print("WARNING: You have a CUDA device, so you should probably run with --gpu")
    
# Set GPU device and backend
if arguments['--gpu']:
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(int(arguments['--gpudev']))

# Set detect anomaly
torch.autograd.set_detect_anomaly(True)

# Parameters
train_path = arguments['--path'] + '/train'
test_path = arguments['--path'] + '/test'
batch_size = int(arguments['--bsize'])
nb_frame = int(arguments['--nbframe'])
if arguments['--o'] == 'None':
    output_dr = os.getcwd() + '/output'
else :
    output_dr = arguments['--o']

    
# load the dataset
if arguments['--inputrep']=="pianoroll":
    dataset = rep_classes.Pianoroll(train_path, nbframe_per_bar=nb_frame) 
    testset = rep_classes.Pianoroll(test_path, nbframe_per_bar=nb_frame)
    input_dim = 128
    seq_length = nb_frame
elif arguments['--inputrep']=="midilike":
    dataset = rep_classes.Midilike(train_path)
    testset = rep_classes.Midilike(test_path)
    input_dim = 1
elif arguments['--inputrep']=="midimono":
    dataset = rep_classes.Midimono(train_path)
    testset = rep_classes.Midimono(test_path)
    input_dim = 1
elif arguments['--inputrep']=="signallike":
    dataset = rep_classes.Signallike(train_path, nbframe_per_bar=nb_frame*2, mono=True)
    testset = rep_classes.Signallike(test_path, nbframe_per_bar=nb_frame*2, mono=True)
    input_dim = dataset.signal_size//64
elif arguments['--inputrep']=="notetuple":
    dataset = rep_classes.Notetuple(train_path)
    testset = rep_classes.NoteTupleRepresentation(test_path)
    input_dim = 5
else :
    raise NotImplementedError("Representation {} not implemented".format(arguments['--inputrep']))
    
# Init the dataloader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1,
                                          pin_memory=True, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=1, 
                                          pin_memory=True, shuffle=True, drop_last=True)

# init writer for tensorboard
writer = SummaryWriter(output_dr + '/runs/' + arguments['--runname'])

# Model parameters
enc_hidden_size = 1024
cond_hidden_size = 1024
dec_hidden_size = 1024
cond_outdim = 512
num_layers_enc = 2
num_layers_dec = 2
num_subsequences = 4
latent_size = 256
if arguments['--inputrep'] in ['pianoroll', 'signallike']:
    output_dim = input_dim
    if arguments['--inputrep']=='pianoroll':
        seq_length = 16
    else :
        seq_length = 64
elif arguments['--inputrep']=="midilike":
    output_dim = len(dataset.vocabulary)
    seq_length = 64
elif arguments['--inputrep']=="midimono":
    output_dim = 130
    seq_length = 16
elif arguments['--inputrep']=="notetuple":
    output_dim = sum([len(v) for v in dataset.vocabs]) + 129
    seq_length = 32

# Instanciate model 
encoder = m.Encoder_RNN(input_dim, enc_hidden_size, latent_size, num_layers_enc)
decoder = m.Decoder_RNN_hierarchical(output_dim, latent_size, cond_hidden_size, 
                                     cond_outdim,dec_hidden_size, num_layers_dec, 
                                     num_subsequences, seq_length)
model = m.VAE(encoder, decoder, arguments['--inputrep'])

# Loss
if arguments['--inputrep'] in ['pianoroll', 'signallike']:
    loss_fn = torch.nn.MSELoss(reduction='sum')
else :
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
 
# Move to GPU
if arguments['--gpu']:
    loss_fn = loss_fn.cuda()
    model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=float(arguments['--lr']))


# Start training
loss_test_min_reconst = 10e6
w_kl = 0
    
for epoch in range(int(arguments['--maxiter'])):
    print('epoch : ' + str(epoch))
    #### Train ####
    loss_mean = 0
    kl_div_mean = 0
    reconst_loss_mean = 0
    nb_pass = 0
    model.train()
    for i, x in tqdm(enumerate(data_loader), total=len(dataset)//batch_size):
        if arguments['--gpu']:
            if arguments['--inputrep'] == "notetuple":
                x[0] = x[0].cuda()
                x[1] = x[1].cuda()
            else :
                x = x.cuda()
        # training pass
        loss, kl_div, reconst_loss = model.batch_pass(x, loss_fn, optimizer, 
                                                      w_kl, dataset)
        loss_mean += loss
        kl_div_mean += kl_div
        reconst_loss_mean += reconst_loss
        nb_pass+=1
        
    # Increase the kl weight
    w_kl = increase_wkl(epoch, w_kl, arguments['--inputrep'])   
    
    #### Test ####
    loss_mean_TEST = 0
    kl_div_mean_TEST = 0
    reconst_loss_mean_TEST = 0
    nb_pass_TEST = 0
    model.eval()
    with torch.no_grad():
        for i, x in tqdm(enumerate(test_loader), total=len(testset)//batch_size):
            if arguments['--gpu']:
                if arguments['--inputrep'] == "notetuple":
                    x[0] = x[0].cuda()
                    x[1] = x[1].cuda()
                else :
                    x = x.cuda()
            # testing pass
            loss, kl_div, reconst_loss = model.batch_pass(x, loss_fn, optimizer, 
                                                          w_kl, dataset, test=True)
            loss_mean_TEST += loss
            kl_div_mean_TEST += kl_div
            reconst_loss_mean_TEST += reconst_loss
            nb_pass_TEST += 1
      
        #### Add to tensorboard ####
        print("adding stuff to tensorboard")
        both_loss = {}
        both_loss['train'] = loss_mean/nb_pass
        both_loss['test'] = loss_mean_TEST/nb_pass_TEST
        writer.add_scalar('data/loss_mean', loss_mean/nb_pass, epoch)
        writer.add_scalar('data/kl_div_mean', kl_div_mean/nb_pass, epoch)
        writer.add_scalar('data/reconst_loss_mean', reconst_loss_mean/nb_pass, epoch)
        writer.add_scalar('data/loss_mean_TEST', loss_mean_TEST/nb_pass_TEST, epoch)
        writer.add_scalar('data/kl_div_mean_TEST', kl_div_mean_TEST/nb_pass_TEST, epoch)
        writer.add_scalar('data/reconst_loss_mean_TEST', reconst_loss_mean_TEST/nb_pass_TEST, epoch)
        writer.add_scalars('data/Losses', both_loss, epoch)
    
    #### Save the model ####
    if arguments['--save']:
        if epoch > 50 and loss_mean_TEST < loss_test_min_reconst:
            loss_test_min_reconst = loss_mean_TEST
            torch.save(model.cpu().state_dict(), 
                       output_dr + '/models/' + arguments['--runname'] + '_epoch_' + str(epoch+1) + '.pth')
            if arguments['--gpu']:
                model = model.cuda()

# End of the script, close the writer
writer.close()


