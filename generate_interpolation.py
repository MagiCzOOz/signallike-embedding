#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:57:30 2020

@author: prang

Usage:
    main.py [-h | --help]
    main.py [--version]
    main.py [--inputrep REP] [--mpath MP] [--dpath DP] [--o OUT] [--nbframe NBFRAME]
            [--start START] [--end END] [--nbpoints POINTS] [--name NAME]
    
Options:
    -h --help  Show this helper
    --version  Show version and exit
    --inputrep REP  Set the representation which will be used as input [default: signallike]
    --mpath MP  The path of the trained model [default: None]
    --dpath DP  The path of the MIDI files folder [default: None]
    --o OUT  Path of the output directory [default: None]
    --nbframe NBFRAME  Number of frames per bar [default: 16]
    --start START  Path of the starting bar [default: None]
    --end END  Path of the ending bar [default: None]
    --nbpoints POINTS  Number of points in the interpolation [default: 24]
    --name NAME  Name of the final MIDI files [default: None]
"""

import representations as rep_classes
import models as m
import random
import torch
import pypianoroll
import numpy as np
import os
from docopt import docopt


# Load arguments from the shell command with docopt
if __name__ == '__main__':
    arguments = docopt(__doc__, version='embedding_generation v1.0')
    print(arguments)

# Global parameters
nb_frame = int(arguments['--nbframe'])
midi_path = arguments['--dpath']
if arguments['--o'] == 'None':
    out_path = os.getcwd() + '/output/interpolations/'
else :
    out_path = arguments['--o']
if arguments['--name'] == 'None':
    name = 'interpolation_' + arguments['--nbpoints']
else :
    name = arguments['--name']

# Load the dataset
if arguments['--inputrep']=='pianoroll':
    dataset = rep_classes.Pianoroll(midi_path, nbframe_per_bar=nb_frame)
else :
    dataset = rep_classes.Signallike(midi_path, nbframe_per_bar=nb_frame*2, mono=False)


# Load the trained model
if arguments['--mpath'] == 'None':
    model_path = os.getcwd() + '/output/models/signallike_JSBchorales.pth'
else :    
    model_path = arguments['--path']

if arguments['--inputrep']=='pianoroll':
    seq_length = 16
    input_dim = 128
else :
    seq_length = 64
    input_dim = dataset.signal_size//64
    
enc_hidden_size = 1024
cond_hidden_size = 1024
dec_hidden_size = 1024
cond_outdim = 512
num_layers_enc = 2
num_layers_dec = 2
num_subsequences = 4
latent_size = 256
output_dim = input_dim

encoder = m.Encoder_RNN(input_dim, enc_hidden_size, latent_size, num_layers_enc)
decoder = m.Decoder_RNN_hierarchical(output_dim, latent_size, cond_hidden_size, 
                                     cond_outdim,dec_hidden_size, num_layers_dec, 
                                     num_subsequences, seq_length)
model = m.VAE(encoder, decoder, arguments['--inputrep'])
model.load_state_dict(torch.load(model_path))
model.eval()


# Get the bars which start and end the interpolation
if arguments['--start'] == 'None':
    ind = random.randint(0,len(dataset))
    starting_point = dataset[ind]
else :    
    starting_point = torch.load(arguments['--start'])

if arguments['--end'] == 'None':
    ind = random.randint(0,len(dataset))
    ending_point = dataset[ind]
else :
    ending_point = torch.load(arguments['--end'])
    
# Get the corresponging latent code
_, _, st_latent, _ = model(starting_point.unsqueeze(0))
_, _, end_latent, _ = model(ending_point.unsqueeze(0))
st_latent = st_latent.squeeze(0).detach().numpy()
end_latent = end_latent.squeeze(0).detach().numpy()


# Interpolate between this two coordinates
latents = torch.tensor(np.linspace(st_latent, end_latent, int(arguments['--nbpoints'])))

# Generate all the bars and concatenate them
for i,latent in enumerate(latents):
    # Generate from it
    generated_bar = model.generate(latent)
    # cleaning
    if arguments['--inputrep'] == 'signallike':
        pr_rec = dataset.back_to_pianoroll(generated_bar.squeeze(0).flatten().detach().numpy())
        pr_rec[pr_rec <= 0.25] = 0
        pr_rec[pr_rec > 0.25] = 64
        y = pr_rec[:,::2]
        for j in [0,4,8,12]:
            y[:,j] = y[:,j+1]
        generated_bar = y.transpose(1,0)
    else :
        generated_bar[generated_bar < 0.8] = 0
        generated_bar[generated_bar >= 0.8] = 64
        generated_bar = generated_bar.squeeze(0).detach().numpy()
    if i == 0:
        progression = generated_bar
    else :
        if not (progression[-16:,:] == generated_bar).all():
            progression = np.concatenate((progression, generated_bar), axis=0)
            
# Use pypianoroll to export it in MIDI format
track = pypianoroll.Track(pianoroll=progression, program=0,
                          is_drum=False, name='Generated_interpolation')
multtrack = pypianoroll.Multitrack(tracks=[track], tempo=90, beat_resolution=4)
pypianoroll.write(multtrack, out_path + name + '.mid')

    

