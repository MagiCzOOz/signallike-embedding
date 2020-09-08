#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:41:24 2020

@author: prang
"""

import torch
from torch import nn
import random



class Encoder_RNN(nn.Module):

    def __init__(self, input_dim, hidden_size, latent_size, num_layers,
                 dropout=0.5, packed_seq=False):
        """ This initializes the encoder """
        super(Encoder_RNN,self).__init__()
        # Parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.packed_seq = packed_seq
        # Layers
        self.RNN = nn.LSTM(input_dim, hidden_size, batch_first=True,
                           num_layers=num_layers, bidirectional=True, 
                           dropout=dropout)


    def forward(self, x, h0, c0, batch_size):
        
        # Pack sequence if needed
        if self.packed_seq:
            x = torch.nn.utils.rnn.pack_padded_sequence(x[0], x[1],
                                                        batch_first=True, 
                                                        enforce_sorted=False)
        # Forward pass
        _, (h, _) = self.RNN(x, (h0, c0))
        # Be sure to not have NaN values
        assert ((h == h).all()),'NaN value in the output of the RNN, try to \
                                lower your learning rate'
        h = h.view(self.num_layers, 2, batch_size, -1)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=1)
        
        return h

    def init_hidden(self, batch_size=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Bidirectional -> num_layers * 2
        return (torch.zeros(self.num_layers * 2, batch_size, self.hidden_size,
                            dtype=torch.float, device=device),) * 2
                            


class Decoder_RNN_hierarchical(nn.Module):

    def __init__(self, input_size, latent_size, cond_hidden_size, cond_outdim, 
                 dec_hidden_size, num_layers, num_subsequences, seq_length,
                 teacher_forcing_ratio=0, dropout=0.5):
        """ This initializes the decoder """
        super(Decoder_RNN_hierarchical, self).__init__()
        # Parameters
        self.num_subsequences = num_subsequences
        self.input_size = input_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.subseq_size = self.seq_length // self.num_subsequences
        # Layers
        self.tanh = nn.Tanh()
        self.fc_init_cond = nn.Linear(latent_size, cond_hidden_size * num_layers)
        self.conductor_RNN = nn.LSTM(latent_size // num_subsequences, cond_hidden_size, 
                                     batch_first=True, num_layers=num_layers, 
                                     bidirectional=False, dropout=dropout)
        self.conductor_output = nn.Linear(cond_hidden_size, cond_outdim)
        self.fc_init_dec = nn.Linear(cond_outdim, dec_hidden_size * num_layers)
        self.decoder_RNN = nn.LSTM(cond_outdim + input_size, dec_hidden_size, 
                                   batch_first=True, num_layers=num_layers, 
                                   bidirectional=False, dropout=dropout)
        self.decoder_output = nn.Linear(dec_hidden_size, input_size)


    def forward(self, latent, target, batch_size, teacher_forcing, device):
        
        # Get the initial state of the conductor
        h0_cond = self.tanh(self.fc_init_cond(latent))
        h0_cond = h0_cond.view(self.num_layers, batch_size, -1).contiguous()
        # Divide the latent code in subsequences
        latent = latent.view(batch_size, self.num_subsequences, -1)
        # Pass through the conductor
        subseq_embeddings, _ = self.conductor_RNN(latent, (h0_cond,)*2)
        subseq_embeddings = self.conductor_output(subseq_embeddings)
        
        # Get the initial states of the decoder
        h0s_dec = self.tanh(self.fc_init_dec(subseq_embeddings))
        h0s_dec = h0s_dec.view(self.num_layers, batch_size, 
                             self.num_subsequences, -1).contiguous()
        # Init the output seq and the first token to 0 tensors
        out = torch.zeros(batch_size, self.seq_length, self.input_size, 
                          dtype=torch.float, device=device)
        token = torch.zeros(batch_size, self.subseq_size, self.input_size,
                            dtype=torch.float, device=device)
        # Autoregressivly output tokens
        for sub in range(self.num_subsequences):
            subseq_embedding = subseq_embeddings[:, sub, :].unsqueeze(1)
            subseq_embedding = subseq_embedding.expand(-1, self.subseq_size, -1)
            h0_dec = h0s_dec[:, :, sub, :].contiguous()
            c0_dec = h0s_dec[:, :, sub, :].contiguous()
            # Concat the previous token and the current sub embedding as input
            dec_input = torch.cat((token, subseq_embedding), -1)
            # Pass through the decoder
            token, (h0_dec, c0_dec) = self.decoder_RNN(dec_input, (h0_dec, c0_dec))
            token = self.decoder_output(token)
            # Fill the out tensor with the token
            out[:, sub*self.subseq_size : ((sub+1)*self.subseq_size), :] = token
            # If teacher_forcing replace the output token by the real one sometimes
            if teacher_forcing:
                if random.random() <= self.teacher_forcing_ratio:
                    token = target[:, sub*self.subseq_size : ((sub+1)*self.subseq_size), :]
                    
        return out
       


class VAE(nn.Module):
    
    def __init__(self, encoder, decoder, input_representation, teacher_forcing=True):
        super(VAE, self).__init__()
        """ This initializes the complete VAE """
        # Parameters
        self.input_rep = input_representation
        self.tf = teacher_forcing
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Layers
        self.hidden_to_mu = nn.Linear(2 * encoder.hidden_size, encoder.latent_size)
        self.hidden_to_sig = nn.Linear(2 * encoder.hidden_size, encoder.latent_size)
        
    def forward(self, x):
        
        if self.input_rep == 'notetuple':
            batch_size = x[0].size(0)
        else :
            batch_size = x.size(0)
        
        # Encoder pass
        h_enc, c_enc = self.encoder.init_hidden(batch_size)
        hidden = self.encoder(x, h_enc, c_enc, batch_size)
        # Reparametrization
        mu = self.hidden_to_mu(hidden)
        sig = self.hidden_to_sig(hidden)
        eps = torch.randn_like(mu).detach().to(self.device)
        latent = (sig.exp().sqrt() * eps) + mu
        
        # Decoder pass
        if self.input_rep == 'midilike':
            # One hot encoding of the target for teacher forcing purpose
            target = torch.nn.functional.one_hot(x.squeeze(2).long(), 
                                                 self.input_size).float()
            x_reconst = self.decoder(latent, target, batch_size,
                                     teacher_forcing=self.tf, device=self.device)
        else :
            x_reconst = self.decoder(latent, x, batch_size,
                                     teacher_forcing=self.tf, device=self.device)
        
        return mu, sig, latent, x_reconst
    
    
    def batch_pass(self, x, loss_fn, optimizer, w_kl, dataset, test=False):
        
        # Zero grad
        self.zero_grad()
    
        # Forward pass
        mu, sig, latent, x_reconst = self(x)
        
        # Compute losses
        kl_div = - 0.5 * torch.sum(1 + sig - mu.pow(2) - sig.exp())
        if self.inputrep in ["midilike", "MVAErep"]:
            reconst_loss = loss_fn(x_reconst.permute(0,2,1), x.squeeze(2).long())
        elif self.inputrep in ["pianoroll", "signallike"]:
            reconst_loss = loss_fn(x_reconst, x)
        elif self.inputrep=="notetuple":
            x_reconst = x_reconst.permute(0,2,1)
            x_in, l = x
            loss_ts_maj = loss_fn(x_reconst[:,:len(dataset.vocabs[0]),:], x_in[:,:,0].long())
            current = len(dataset.vocabs[0])
            loss_ts_min = loss_fn(x_reconst[:,current:current+len(dataset.vocabs[1]),:], x_in[:,:,1].long())
            current += len(dataset.vocabs[1])
            loss_pitch = loss_fn(x_reconst[:,current:current + 129,:], x_in[:,:,2].long())
            current += 129
            loss_dur_maj = loss_fn(x_reconst[:,current:current+len(dataset.vocabs[2]),:], x_in[:,:,3].long())
            current += len(dataset.vocabs[2])
            loss_dur_min = loss_fn(x_reconst[:,current:current+len(dataset.vocabs[3]),:], x_in[:,:,4].long())
            reconst_loss = loss_ts_maj + loss_ts_min + loss_pitch + loss_dur_maj + loss_dur_min
        
        # Backprop and optimize
        if not test:
            loss = reconst_loss + (w_kl * kl_div)
            loss.backward()     
            optimizer.step()
        else :
            loss = reconst_loss + kl_div
        
        return loss, kl_div, reconst_loss
        
    
    def generate(self, latent):
        
        # Create dumb target
        input_shape = (1, self.decoder.seq_length, self.decoder.input_size)
        db_trg = torch.zeros(input_shape)
        # Forward pass in the decoder
        generated_bar = self.decoder(latent.unsqueeze(0), db_trg, batch_size = 1,
                                     device=self.device, teacher_forcing=False)
        
        return generated_bar 
