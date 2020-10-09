# Musical symbolic representations and embeddings

This is the support page for the paper "Signal-domain representation of symbolic music for learning embedding spaces" published in [the 2020 Joint Conference on AI Music Creativity](https://boblsturm.github.io/aimusic2020/).

## Synthetic dataset

In order to precisely evaluate how embedding spaces are able to capture musical theory principles, we design a principled generative approach. This allows to obtain large sets of evaluation data with controlled properties, while following known music theory rules. For our experiments, we apply this approach to the four-part harmonized chorales written by J-S Bach. Hence, we produce a dataset composed by bars of synthetic chorales *C major* or *a minor* tonality. We follow the strict modulation rules implemented by Bach in his chorales and allow only the use of neighbouring tonalities (*Gmaj*, *Fmaj*, *emin*, *dmin* in our case). Bars are either in one of these six tonalities or modulate between them. 

We generate the data by random sampling with the following procedure :
1. We generate sequences of tonal functions with first-order Markov chains, with transition probabilities defined by expert composers.
2. We expand this sequence as a four-voices realisation, by randomly picking chords in major triads, minor triads, diminished triads and dominant sevenths. This defines a *skeleton* of only quarter notes.
3. Based on the *skeleton*, we generate more sophisticated realisations by using passing tones, neighboring tones, suspensions and retardations without fundamental changes in the chord progression.

In order to further control the generation properties, we define two set of rules related to the harmony and voices progressions. These rules allow to :
1. Define prior constraints on sampling through
    * voice pitch ranges
    * rules on double and missing notes in chords
2. Exclude erroneous samples containing
    * bad transitions of the seventh and leading-tone
    * parallel octaves, fifths and unison
    * direct octaves, fifths and unison
    * unisons left by direct motion
    * unisons approached by oblique motion
  
Finally, we obtain a set of 21966 realisations from 370 different skeletons where the links between each realisation and its corresponding skeleton have been kept. In our experiments, this corpus is used only for evaluation purpose. Hence, none of these data are used during the training.

## Usage

### Train

You can train the model on your own dataset by using the `train.py` function with the following options :

    Usage:
      train.py [-h | --help]
      train.py [--version]
      train.py [--gpu] [--gpudev GPUDEVICE] [--lr LR] [--maxiter MITER]
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
      --path P  The path of the MIDI files folder (with a test and train folder)
      --bsize BSIZE  Batch size [default: 16]
      --nbframe NBFRAME  Number of frames per bar [default: 16]
      --o OUT  Path of the output directory [default: None]
      --save  Save the models during the training or not [default: True]

The possible values for the input representations are *signallike*, *pianoroll*, *midilike*, *notetuple* and *midimono*. See the code in `representations.py` for the implementation details of each representations.

Your MIDI files folder have to contained a *train* and a *test* folder where all the files have been splitted at your convenience.

The `--nbframe` option is only used for the signallike and the pianoroll representations.

### Generate

You can generate interpolation between two points in the learned latent space by using the `generate_interpolation.py` function with the following options :

    Usage:
        generate_interpolation.py [-h | --help]
        generate_interpolation.py [--version]
        generate_interpolation.py [--inputrep REP] [--mpath MP] [--dpath DP] [--o OUT] [--nbframe NBFRAME]
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

If the options `--start` and `--end` are not provided, the starting and ending points are chosen randomly among all the dataset.
