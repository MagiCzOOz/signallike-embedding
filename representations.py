#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:28:14 2020

@author: prang
"""

import torch
import pretty_midi
import os
import numpy as np
from operator import attrgetter
from bisect import bisect_left
import librosa
from abc import ABC, abstractmethod

#%%

# MIDI extensions
EXT = ['.mid', '.midi', '.MID', '.MIDI']
# Primes number
PRIMES = [43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
          109, 113, 127, 131, 137, 149, 157, 163, 167, 173, 179, 191, 197,
          211, 223, 227, 233, 239, 251, 257, 263, 269, 277, 281, 293, 307,
          311, 317, 331, 337, 347, 353, 359, 367, 373, 379, 383, 389, 397,
          401, 409, 419, 431, 439, 443, 449, 457, 461, 467, 479, 487, 491,
          499, 503, 509, 521, 541, 547, 557, 563, 569, 577, 587, 593, 599,
          607, 613, 617, 631, 641, 647, 653, 659, 673, 677, 683, 691, 701,
          709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797,
          809, 821, 827, 839, 853, 857, 863, 877, 881, 887, 907, 911, 919,
          929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
          1019, 1031, 1039, 1049, 1061, 1069, 1087, 1091, 1097, 1103, 1109,
          1117, 1123, 1129, 1151, 1163, 1171, 1181, 1187, 1193, 1201, 1213,
          1217, 1223, 1229, 1237, 1249, 1259, 1277, 1283, 1289, 1297, 1301,
          1307, 1319, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427,
          1433, 1439, 1447, 1451, 1459, 1471, 1481, 1487, 1493, 1499, 1511,
          1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597,
          1601, 1607, 1613, 1619, 1627, 1637, 1657, 1663, 1667, 1693, 1697,
          1709, 1721, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1801,
          1811, 1823, 1831, 1847, 1861, 1867, 1871, 1877, 1889, 1901, 1907,
          1913, 1931, 1949, 1973, 1979, 1987, 1993, 1997, 2003, 2011, 2017,
          2027, 2039, 2053, 2063]
                

# Usefull functions
def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before


# Abstract class for the different input representations
class Representation(ABC):
    
    def __init__(self, root_dir, nbframe_per_bar=16, mono=False, export=False):
        """
        Args:
            root_dir (string) : Path of the directory with all the MIDI files
            nbframe_per_bar (int) : Number of frame contained in a bar
            export (bool) : Force the bar to be exported in .pt file or not
        """
        assert any((fname.endswith(tuple(EXT))) for fname in os.listdir(root_dir)), "There are no MIDI files in %s" % root_dir
        # root directory path which contains the files
        self.rootdir = root_dir
        # midi files names
        self.midfiles = [fname for fname in os.listdir(root_dir) if (fname.endswith(tuple(EXT)))]
        # Number of frame per bar
        self.nbframe_per_bar = nbframe_per_bar
        # Force export or not
        self.export = export
        # Monophonic data (separate voices)
        self.mono = mono
        # number of tracks contained in the dataset
        self.nb_tracks = len(self.midfiles)
    
    
    
    def __len__(self):
        
        return self.nb_bars
   
                      
    
    def __getitem__(self, index):

        return torch.load(self.prbar_path + '/' + self.barfiles[index])
    
                    
    @abstractmethod                                    
    def per_bar_export(self):
        """
        This function take all the midi files, load them into a pretty_midi object.
        For a complete documentation of pretty_midi go to :
            http://craffel.github.io/pretty-midi/
        
        The midi file is then processed to obtain the given representation of each bar with
        Finally, it will export each of theses bars in a separate .pt
        """
        pass
    
    

################################# PIANO-ROLL #################################   
        
class Pianoroll(Representation):
    
    def __init__(self, root_dir, nbframe_per_bar=16, mono=False, export=False):
        super().__init__(root_dir, nbframe_per_bar=nbframe_per_bar, mono=mono, export=export)
        # Path witch contains the sliced piano-roll
        if mono:
            self.prbar_path = root_dir + "/pianoroll_bar_mono_" + str(self.nbframe_per_bar)
        else :
            self.prbar_path = root_dir + "/pianoroll_bar_" + str(self.nbframe_per_bar)
        if not os.path.exists(self.prbar_path):
            try:  
                os.mkdir(self.prbar_path)
            except OSError:  
                print ("Creation of the directory %s failed" % self.prbar_path)
            else:  
                print ("Successfully created the directory %s " % self.prbar_path)
            # Export the piano-roll bat
            self.per_bar_export()
        else :
            if export:
                self.per_bar_export()
        # .pt files names
        self.barfiles = [fname for fname in os.listdir(self.prbar_path) if fname.endswith('.pt')]
        # total number of bars
        self.nb_bars = len(self.barfiles)
        
        

    def sliced_and_save_pianoroll(self, pianoroll, downbeats, fs, num_bar):
        
        for i in range(len(downbeats)-1):
            sp = pianoroll[:,int(round(downbeats[i]*fs)):int(round(downbeats[i+1]*fs))-1]
            if sp.shape[1] > 256:
                sp = sp[:,0:256]
            elif sp.shape[1] < 256 and sp.shape[1] > 0:
                sp = np.pad(sp,((0,0),(0,256 - sp.shape[1])), 'edge')
            if sp.shape[1] > 0 :
                # downsample
                sp = sp[:,::int(256/self.nbframe_per_bar)]
                # convert to tensor
                sp = torch.Tensor(sp)
                assert (sp.shape[1]==self.nbframe_per_bar), "Error, a piano-roll have the wrong size : %s" % sp.shape[1]
                # binarize
                sp[sp!=0]=1
                # Save the tensor
                torch.save(sp.permute(1,0), self.prbar_path + "/prbar" + str(num_bar) +  ".pt")
                num_bar += 1
        
        return num_bar
    


    def per_bar_export(self):

        num_error = 0
        num_bar = 0
        # load each .mid file in a pretty_midi object
        for index in range(len(self.midfiles)):
            try:
                midi_data = pretty_midi.PrettyMIDI(self.rootdir + '/' + self.midfiles[index])
                downbeats = midi_data.get_downbeats()
                fs = 257 / (midi_data.get_end_time() / len(downbeats))
                # If monophonic data is required, we separate each voice
                if self.mono:
                    for inst in midi_data.instruments:
                        if not inst.is_drum:
                            pianoroll = inst.get_piano_roll(fs=fs)
                            num_bar = self.sliced_and_save_pianoroll(pianoroll, downbeats, fs, num_bar)
                else :
                    pianoroll = midi_data.get_piano_roll(fs=fs)
                    num_bar = self.sliced_and_save_pianoroll(pianoroll, downbeats, fs, num_bar)
            except KeyError:
                num_error += 1
        print("total number of file : ", len(self.midfiles))
        print('num error : ', num_error)
        
        

################################# MIDI-like ################################## 

class Midilike(Representation):
    
    def __init__(self, root_dir, nbframe_per_bar=16, mono=False, export=False):
        super().__init__(root_dir, nbframe_per_bar=nbframe_per_bar, mono=mono, export=export)
        # One hot encoding of the vocabulary
        self.vocabulary = self.get_vocab_encoding()
        # Path witch contains the sliced piano-roll
        self.prbar_path = root_dir + "/MIDIlike_bar"
        if not os.path.exists(self.prbar_path):
            try:  
                os.mkdir(self.prbar_path)
            except OSError:  
                print ("Creation of the directory %s failed" % self.prbar_path)
            else:  
                print ("Successfully created the directory %s " % self.prbar_path)
            # Export the piano-roll bat
            self.per_bar_export()
        else :
            if export:
                self.per_bar_export()
        # .pt files names
        self.barfiles = [fname for fname in os.listdir(self.prbar_path) if fname.endswith('.pt')]
        # total number of bars
        self.nb_bars = len(self.barfiles)
             
        
                
    def get_vocab_encoding(self):
        """
        Return a dictionnary with the corresponding indexes of every word contained in
        the vocabulary (one hot encoding).
        
        e.g : vocab = {'NOTE_ON<1>' : 0
                       'NOTE_ON<2>' : 1
                           ...
                       'TIME_SHIFT<1> : 128
                           ...             }
        """
        vocab = {}
        current_ind = 0
        
        # All the NOTE_ON events
        rootstr = "NOTE_ON<"
        for i in range(0, 128):
            event = rootstr + str(i) + '>'
            vocab[event] = current_ind
            current_ind += 1
        # All the NOTE_OFF events
        rootstr = "NOTE_OFF<"
        for i in range(0, 128):
            event = rootstr + str(i) + '>'
            vocab[event] = current_ind
            current_ind += 1
        # All the TIME_SHIFT events
        rootstr = "TIME_SHIFT<"
        for i in range(10, 1001, 10):
            event = rootstr + str(i) + '>'
            vocab[event] = current_ind
            current_ind += 1
        # All the SET_VELOCITY events
        rootstr = "SET_VELOCITY<"
        for i in range(0,128,4):
            event = rootstr + str(i) + '>'
            vocab[event] = current_ind
            current_ind += 1
        # The NOTHING event
        vocab['NOTHING'] = current_ind
        
        return vocab
    
    
    
    def string_representation(self, Vinst):
        """
        Return the representation with string ("NOTE_ON<56>, ...) from the 
        corresponding integer representation Vinst (list of int)
        """
        str_rep = []
        for i in Vinst:
            str_rep.append(list(self.vocabulary.keys())[list(self.vocabulary.values()).index(int(i))])
        
        return str_rep
    
    
    
    def per_bar_export(self):
        """
        This function take all the midi files, load them into a pretty_midi object.
        For a complete documentation of pretty_midi go to :
            http://craffel.github.io/pretty-midi/
        
        The midi file is then processed to obtain a MIDI-like event-based representation.
        More info on this representation here : 
            https://arxiv.org/pdf/1809.04281.pdf
        
        Ex :  SET_VELOCITY<80>, NOTE_ON<60>
              TIME_SHIFT<500>, NOTE_ON<64>
              TIME_SHIFT<500>, NOTE_ON<67>
              TIME_SHIFT<1000>, NOTE_OFF<60>, NOTE_OFF<64>, NOTE_OFF<67>
              TIME_SHIFT<500>, SET_VELOCITY<100>, NOTE_ON<65>
              TIME_SHIFT<500>, NOTE_OFF<65>
              
        Finally, it will export each of theses bars in a separate .pt
        """
        # number of error with the key analyzer
        num_error = 0
        # array of all the tensor representing each bar
        all_bars = []
        # load each .mid file in a pretty_midi object
        for index in range(len(self.midfiles)):
            try:
                midi_data = pretty_midi.PrettyMIDI(self.rootdir + '/' + self.midfiles[index])
                downbeats = midi_data.get_downbeats() #start_time = midi_data.estimate_beat_start()
                current_velocity = 64.
                # Possible value for the velocity
                velocity_list = [i for i in range(0,128,4)]
                # Possible value for the time shifts
                timeshift_list = [i for i in range(10,1001,10)]
                for i in range(len(downbeats)-1):
                    list_notes = []
                    V = []
                    for inst in midi_data.instruments:
                        if not inst.is_drum:
                            for n in inst.notes:
                                if (n.start < downbeats[i+1] and n.end >= downbeats[i]):
                                    list_notes.append(n)
                    # Sort list by pitch
                    list_notes = sorted(list_notes, key = lambda i: i.pitch, reverse = False)
                    if len(list_notes)==0:
                        gap = (downbeats[i+1] - downbeats[i])*1000
                        while(gap > 1000):
                            V.append(self.vocabulary['TIME_SHIFT<1000>'])
                            gap = gap - 1000    
                        timeshift = takeClosest(timeshift_list, gap)
                        V.append(self.vocabulary['TIME_SHIFT<' + str(timeshift) + '>'])
                    else:
                        # iterate over list_notes to construct the representation
                        current_time = downbeats[i]
                        while(list_notes):
                            closest_note_on = min(list_notes, key=attrgetter('start'))
                            closest_note_off = min(list_notes, key=attrgetter('end'))
                            if closest_note_off.end > closest_note_on.start:
                                gap = (closest_note_on.start - current_time)*1000
                                if gap > timeshift_list[0]/2:
                                    while(gap > 1000):
                                        V.append(self.vocabulary['TIME_SHIFT<1000>'])
                                        gap = gap - 1000    
                                    timeshift = takeClosest(timeshift_list, gap)
                                    V.append(self.vocabulary['TIME_SHIFT<' + str(timeshift) + '>'])
                                if takeClosest(velocity_list, closest_note_on.velocity) != current_velocity:
                                    veloc = takeClosest(velocity_list, closest_note_on.velocity)
                                    V.append(self.vocabulary['SET_VELOCITY<' + str(veloc) + '>'])
                                    current_velocity = veloc
                                V.append(self.vocabulary['NOTE_ON<' + str(closest_note_on.pitch) + '>'])
                                if closest_note_on.start > current_time:
                                    current_time = closest_note_on.start
                                if closest_note_on.end > downbeats[i+1]:
                                    list_notes.remove(closest_note_on)
                                else :
                                    # Set a value > end to start to not taking it in account anymore 
                                    closest_note_on.start = closest_note_on.end + 10
                            else :
                                gap = (closest_note_off.end - current_time)*1000
                                if gap > timeshift_list[0]/2:
                                    while(gap > 1000):
                                        V.append(self.vocabulary['TIME_SHIFT<1000>'])
                                        gap = gap - 1000
                                    timeshift = takeClosest(timeshift_list, gap)
                                    V.append(self.vocabulary['TIME_SHIFT<' + str(timeshift) + '>'])
                                V.append(self.vocabulary['NOTE_OFF<' + str(closest_note_off.pitch) + '>'])
                                current_time = closest_note_off.end
                                list_notes.remove(closest_note_off)
                    # Store the tensor in all_bars
                    all_bars.append(torch.tensor(V))
            except KeyError:
                num_error += 1
        print('num error : ', num_error)
        # Cleaning of the tensor : supressing ones with more than 160 events
        # and padding to have a constant size equal to 160
        empty_bar = False
        total_num = 0
        for i, vec in enumerate(all_bars):
            # add the empty bar only one time
            if len(vec) == 1:
                if not empty_bar:
                    clean_vec = torch.tensor([self.vocabulary['NOTHING']]*64)
                    clean_vec[0] = vec
                    torch.save(clean_vec.unsqueeze(1), self.prbar_path + "/Mlikebar_" + str(i) + ".pt")
                    empty_bar = True
                    total_num += 1
            elif len(vec) < 64:
                clean_vec = torch.tensor([self.vocabulary['NOTHING']]*64)
                clean_vec[:len(vec)]=vec
                torch.save(clean_vec.unsqueeze(1), self.prbar_path + "/Mlikebar_" + str(i) + ".pt")
                total_num += 1
            elif len(vec) == 64:
                torch.save(vec.unsqueeze(1), self.prbar_path + "/Mlikebar_" + str(i) + ".pt")
                total_num += 1
        print("Initial number of bar : {}\n \
               After cleaning : {}\n \
               Number of suppression : {}".format(len(all_bars), total_num, len(all_bars) - total_num))
        
        


############################### MIDI-like mono ############################### 
        
class Midimono(Representation):
    
    def __init__(self, root_dir, nbframe_per_bar=16, mono=True, export=False):
        super().__init__(root_dir, nbframe_per_bar=nbframe_per_bar, mono=mono, export=export)
        # Path witch contains the sliced piano-roll
        self.prbar_path = root_dir + "/MIDIMono_bar"
        if not os.path.exists(self.prbar_path):
            try:  
                os.mkdir(self.prbar_path)
            except OSError:  
                print ("Creation of the directory %s failed" % self.prbar_path)
            else:  
                print ("Successfully created the directory %s " % self.prbar_path)
            # Export the piano-roll bat
            self.per_bar_export()
        else :
            if export:
                self.per_bar_export()
        # .pt files names
        self.barfiles = [fname for fname in os.listdir(self.prbar_path) if fname.endswith('.pt')]
        # total number of bars
        self.nb_bars = len(self.barfiles)
        
    
    
    def get_polyphonic_bars(self, pr_dataset):
        
        indices = set()
        for i in range(len(pr_dataset)):
            for j,frame in enumerate(pr_dataset[i]):
                if frame.nonzero().nelement() > 1:
                    indices.add(i)
                    
        return indices
    
                
    
    def to_pianoroll(self, v):
        
        pianoroll = torch.zeros(16, 128)
        current_note = -1
        for i,e in enumerate(v):
            if e < 128:
                pianoroll[i, int(e)] = 1
                current_note = int(e)
            elif e == 128:
                if current_note != 129:
                    pianoroll[i, current_note] = 1
                    
        return pianoroll
        
    
    
    def per_bar_export(self):
        
        PR = Pianoroll(self.rootdir, nbframe_per_bar=16, mono=True)
        poly_bars = self.get_polyphonic_bars(PR)
        num_vec = 0
        for i in range(len(PR)):
            if i not in poly_bars:
                vec = torch.zeros(16)
                current_note = -1
                for j,frame in enumerate(PR[i]):
                    if frame.nonzero().nelement() == 0:
                        if current_note != 129 and current_note != -1:
                            # note_off event
                            vec[j] = 129
                            current_note = 129
                        else :
                            # rest event
                            vec[j] = 128
                            if current_note == -1:
                                current_note = 129
                    else :
                        if current_note == int(frame.nonzero()):
                            # rest event
                            vec[j] = 128
                        else :
                            # note_on event
                            vec[j] = int(frame.nonzero())
                            current_note = int(frame.nonzero())
                # Save the tensor
                torch.save(vec.unsqueeze(1), self.prbar_path + "/MVAEbar_" + str(num_vec) + ".pt")
                num_vec += 1
                
                

################################# NoteTuple ##################################                
 
class Notetuple(Representation):
    
    def __init__(self, root_dir, nbframe_per_bar=16, mono=False, export=False):
        super().__init__(root_dir, nbframe_per_bar=nbframe_per_bar, mono=mono, export=export)
        # vocabs
        self.vocabs = self.get_vocabs_encoding()
        self.ts_major = self.vocabs[0]
        self.ts_minor = self.vocabs[1]
        self.dur_major = self.vocabs[2]
        self.dur_minor = self.vocabs[3]
        # Path witch contains the sliced piano-roll
        self.prbar_path = root_dir + "/NoteTuple_bar"
        if not os.path.exists(self.prbar_path):
            try:  
                os.mkdir(self.prbar_path)
            except OSError:  
                print ("Creation of the directory %s failed" % self.prbar_path)
            else:  
                print ("Successfully created the directory %s " % self.prbar_path)
            # Export the bar
            self.per_bar_export()
        else :
            if export:
                self.per_bar_export()
        # .pt files names
        self.barfiles = [fname for fname in os.listdir(self.prbar_path) if fname.endswith('.pt')]
        # total number of bars
        self.nb_bars = len(self.barfiles)
        
        
        
    def get_vocabs_encoding(self):
        # timeshift major_ticks_vocab
        ts_major = {}
        ind = 0
        for val in [i for i in range(0,9601,800)]:
            ts_major[val] = ind
            ind += 1
        ts_major[-1] = ind
        # timeshift minor_ticks_vocab
        ts_minor = {}
        ind = 0
        for val in [i for i in range(0,800,10)]:
            ts_minor[val] = ind
            ind += 1
        ts_minor[-1] = ind
        # duration major_ticks_vocab
        dur_major = {}
        ind = 0
        for val in [i for i in range(0,9501,500)]:
            dur_major[val] = ind
            ind += 1
        dur_major[-1] = ind
        # duration minor_ticks_vocab
        dur_minor = {}
        ind = 0
        for val in [i for i in range(0,500,10)]:
            dur_minor[val] = ind
            ind += 1
        dur_minor[-1] = ind
        return ts_major, ts_minor, dur_major, dur_minor
        
    
    
    def value_to_class(self, bar):
        # Change the value of the timeshift and duration to a class number
        for i,tupl in enumerate(bar):
            for j,v in enumerate(tupl):
                if j == 0:
                    bar[i][j] = self.ts_major[int(v)]
                if j == 1:
                    bar[i][j] = self.ts_minor[int(v)]
                if j == 2:
                    if v == -1:
                       bar[i][j] = 128 
                if j == 3:
                    bar[i][j] = self.dur_major[int(v)]
                if j == 4:
                    bar[i][j] = self.dur_minor[int(v)]
        return bar
    
    
    
    def class_to_value(self, bar):
        # Change the class number of the timeshift and duration to the real value
        for i,tupl in enumerate(bar):
            for j,v in enumerate(tupl):
                if j == 0:
                    bar[i][j] = list(self.ts_major.keys())[list(self.ts_major.values()).index(int(v))]
                if j == 1:
                    bar[i][j] = list(self.ts_minor.keys())[list(self.ts_minor.values()).index(int(v))]
                if j == 2:
                    if v == 128:
                        bar[i][j] = -1
                if j == 3:
                    bar[i][j] = list(self.dur_major.keys())[list(self.dur_major.values()).index(int(v))]
                if j == 4:
                    bar[i][j] = list(self.dur_minor.keys())[list(self.dur_minor.values()).index(int(v))]
        return bar                
        
    
    
    def per_bar_export(self):
        num_error = 0
        # to store all the bars
        all_bars = [] 
        for index in range(len(self.midfiles)):
            try:
                # load each .mid file in a pretty_midi object
                midi_data = pretty_midi.PrettyMIDI(self.rootdir + '/' + self.midfiles[index])
                downbeats = midi_data.get_downbeats() #start_time = midi_data.estimate_beat_start()
                # Possible value for the time shifts (from 0 to 10s)
                # 13 major ticks
                timeshift_major_ticks = [i for i in range(0,9601,800)]
                # 77 minor ticks
                timeshift_minor_ticks = [i for i in range(0,800,10)]
                # Possible value for the duration
                # 13 major ticks
                dur_major_ticks = [i for i in range(0,9501,500)]
                # 77 minor ticj=ks
                dur_minor_ticks = [i for i in range(0,500,10)]
                for i in range(len(downbeats)-1):
                    list_notes = []
                    V = []
                    for inst in midi_data.instruments:
                        if not inst.is_drum:
                            for n in inst.notes:
                                if (n.start < downbeats[i+1] and n.start >= downbeats[i]):
                                    list_notes.append(n)
                    list_notes = sorted(list_notes, key = lambda i: i.pitch, reverse = False)
                    # iterate over list_notes to construct the representation
                    current_time = downbeats[i]
                    while(list_notes):
                        closest_note_on = min(list_notes, key=attrgetter('start'))
                        time_shift = (closest_note_on.start - current_time)*1000
                        tmat = timeshift_major_ticks[int(time_shift//800)]
                        tmit = timeshift_minor_ticks[int((time_shift%800)//10)]
                        duration = (closest_note_on.end - closest_note_on.start)*1000
                        dmat = dur_major_ticks[int(duration//500)]
                        dmit = dur_minor_ticks[int((duration%500)//10)]
                        current_time = closest_note_on.start
                        V.append((tmat,tmit,closest_note_on.pitch, dmat, dmit))
                        list_notes.remove(closest_note_on)
                    # Store the tensor in all_bars
                    all_bars.append(torch.tensor(V))
            except KeyError:
                num_error += 1
        print('num error : ', num_error)   
        # Save all tensor
        total_num = 0
        for i, vec in enumerate(all_bars):
            if len(vec) < 32 and len(vec) > 0:
                clean_vec = torch.zeros(32,5).fill_(-1)
                clean_vec[:len(vec)] = vec
                clean_vec = self.value_to_class(clean_vec)
                torch.save((clean_vec, len(vec)), self.prbar_path + "/Ntuplebar" + str(i) + ".pt")
                total_num += 1
            elif len(vec) == 32:
                vec = self.value_to_class(vec)
                torch.save((clean_vec, len(vec)), self.prbar_path + "/Ntuplebar" + str(i) + ".pt")
                total_num += 1
        print("Initial number of bar : {}\n \
               After cleaning : {}\n \
               Number of suppression : {}".format(len(all_bars), total_num, len(all_bars) - total_num)) 




################################# Signal-like ################################
               
class Signallike(Representation):
    
    def __init__(self, root_dir, nbframe_per_bar=16, mono=False, export=False):
        super().__init__(root_dir, nbframe_per_bar=nbframe_per_bar, mono=mono, export=export)
        # Path to export the .pt files
        if self.mono:
            self.prbar_path = root_dir + "/Signallike_bar_mono_" + str(self.nbframe_per_bar)
        else:
            self.prbar_path = root_dir + "/Signallike_bar_" + str(self.nbframe_per_bar)
        if not os.path.exists(self.prbar_path):
            try:  
                os.mkdir(self.prbar_path)
            except OSError:  
                print ("Creation of the directory %s failed" % self.prbar_path)
            else:  
                print ("Successfully created the directory %s " % self.prbar_path)
            # Export the piano-roll bat
            self.per_bar_export()
        else :
            if export:
                self.per_bar_export()
        # .pt files names
        self.barfiles = [fname for fname in os.listdir(self.prbar_path) if fname.endswith('.pt')]
        # total number of bars
        self.nb_bars = len(self.barfiles)
        # Size of the signal representation
        self.signal_size = len(self.__getitem__(0).flatten())
        
        
    
    def back_to_pianoroll(self, V):
        """
        Inverse the process : get a piano-roll from a signal-like representation V.
        """
        PR = ((np.abs(librosa.core.stft(V , n_fft=2048, window='blackman'))))[PRIMES[:128]]
        return abs(PR)
    
    
    
    def get_polyphonic_bars(self, pr_dataset):
        indices = set()
        for i in range(len(pr_dataset)):
            for j,frame in enumerate(pr_dataset[i]):
                if frame.nonzero().nelement() > 1:
                    indices.add(i)
        return indices
    
    
    
    def per_bar_export(self):
        """
        This function take the self.midfiles[index], load it into a pretty_midi object.
        For a complete documentation of pretty_midi go to :
            http://craffel.github.io/pretty-midi/ 
       
        The midi file is then processed with stft to obtain a signal-like representation
        and exported in a .pt file.
        """
        PR = Pianoroll(self.rootdir, nbframe_per_bar=self.nbframe_per_bar, mono=self.mono)
        if self.mono:
            poly_bars = self.get_polyphonic_bars(PR)
        else:
            poly_bars = []
        for i in range(len(PR)):
            if i not in poly_bars:
                final_vals = np.zeros((1025, PR[i].permute(1,0).shape[1])).astype(complex)
                final_vals[PRIMES[:128], :] = np.array(PR[i].permute(1,0)) + 1j * ((np.array(PR[i].permute(1,0)) > 0))
                V = torch.Tensor(librosa.core.istft(final_vals, window='blackman'))
                torch.save(V.reshape(64, -1), self.prbar_path + "/Slikebar_" + str(i) + ".pt")
