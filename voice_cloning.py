from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch

import soundfile as sf
import pyrubberband as pyrb

import re
import spacy
from spacy.pipeline import Sentencizer


# Prepare NLP pipeline
nlp = spacy.load("en_core_web_sm", disable=["tagger","parser", "ner"])
sentencizer = Sentencizer(punct_chars=[".", "?", "!", ":", "..."])
nlp.add_pipe(sentencizer)

"""#### Prepare the models"""

# Print some environment information (for debugging purposes)

    ## Print some environment information (for debugging purposes)
print("Running a test of your configuration...\n")
if not torch.cuda.is_available():
    print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
          "for deep learning, ensure that the drivers are properly installed, and that your "
          "CUDA version matches your PyTorch installation. CPU-only inference is currently "
          "not supported.", file=sys.stderr)
    quit(-1)
device_id = torch.cuda.current_device()
gpu_properties = torch.cuda.get_device_properties(device_id)
print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
      "%.1fGb total memory.\n" %
      (torch.cuda.device_count(),
        device_id,
        gpu_properties.name,
        gpu_properties.major,
        gpu_properties.minor,
        gpu_properties.total_memory / 1e9))

#  Set location of pretrained models

enc_model_fpath = Path("encoder/saved_models/pretrained.pt")
syn_model_dir = Path("synthesizer/saved_models/logs-pretrained/")
voc_model_fpath = Path("vocoder/saved_models/pretrained/pretrained.pt")

# Load the models one by one


## Load the models one by one.
print("Preparing the encoder, the synthesizer and the vocoder...")
# encoder.load_model(args.enc_model_fpath)
# synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
# vocoder.load_model(args.voc_model_fpath)

encoder.load_model(enc_model_fpath)
synthesizer = Synthesizer(syn_model_dir.joinpath("taco_pretrained"))
vocoder.load_model(voc_model_fpath)

# Run a test

print("Testing your configuration with small inputs.")
# Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
# sampling rate, which may differ.
# If you're unfamiliar with digital audio, know that it is encoded as an array of floats
# (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
# The sampling rate is the number of values (samples) recorded per second, it is set to
# 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
# to an audio of 1 second.
print("\tTesting the encoder...")
encoder.embed_utterance(np.zeros(encoder.sampling_rate))

# Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
# returns, but here we're going to make one ourselves just for the sake of showing that it's
# possible.
embed = np.random.rand(speaker_embedding_size)
# Embeddings are L2-normalized (this isn't important here, but if you want to make your own
# embeddings it will be).
embed /= np.linalg.norm(embed)
# The synthesizer can handle multiple inputs with batching. Let's create another embedding to
# illustrate that
embeds = [embed, np.zeros(speaker_embedding_size)]
print(embeds)
text_arr = ["test 1", "test 2"]
print("\tTesting the synthesizer... (loading the model will output a lot of text)")
mels = synthesizer.synthesize_spectrograms(text_arr, embeds)

# The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We
# can concatenate the mel spectrograms to a single one.
mel = np.concatenate(mels, axis=1)
# The vocoder can take a callback function to display the generation. More on that later. For
# now we'll simply hide it like this:
no_action = lambda *args: None
print("\tTesting the vocoder...")
# For the sake of making this test short, we'll pass a short target length. The target length
# is the length of the wav segments that are processed in parallel. E.g. for audio sampled
# at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
# 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
# that has a detrimental effect on the quality of the audio. The default parameters are
# recommended in general.
vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

print("All test passed! You can now synthesize speech.\n\n")

"""#### Interactive part

Interactive speech generation
"""

# Compute the embedding
num_generated = 0
# Get the reference audio filepath
message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
          "wav, m4a, flac, ...):\n"
in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))
print(in_fpath)

# First, we load the wav using the function that the speaker encoder provides. This is
# important: there is preprocessing that must be applied.

# The following two methods are equivalent:
# - Directly load from the filepath:
preprocessed_wav = encoder.preprocess_wav(in_fpath)
# - If the wav is alibrosaeady loaded:
original_wav, sampling_rate = librosa.load(in_fpath)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
print("Loaded file succesfully")

# Then we derive the embedding. There are many functions and parameters that the
# speaker encoder interfaces. These are mostly for in-depth research. You will typically
# only use this function (with its default parameters):
embed = encoder.embed_utterance(preprocessed_wav)
print("Created the embedding")

"""##### This is the fun part

Text scratchpad:

Now listen here jack now wait just a minute we are going to cure cancer you know. When i was your age no really im serious i am serious when I was your age when I was your age. We used to have to go out in the woods and buy our own candy from the nickel store you know, I mean I had one or two friends here and there, I did not have a black friend. When I was a single dad, you know, my son, he died, my son died in a rack fighting the weapons of mass destruction.
"""

!mkdir output

# Generating the sentences
text = input("Insert a text to be synthesized:\n")
document = nlp(text)

outfiles = []
for sent in list(document.sents):
    cleaned_string = ""
    removed_chars = ['[\d+]', '\[',  '\]', '“', '”', '’', '\(', '\)']
    for token in sent:
      checked_token = re.sub(' —', ',', checked_token)
      checked_token = re.sub('—', '', checked_token)
      checked_token = re.sub(':', '.', checked_token)
      checked_token = re.sub(';', ',', checked_token)
      checked_token = re.sub('\.\.\.', '.', checked_token)
      for char in removed_chars:
          checked_token = re.sub(char, '', token.string)
      cleaned_string += checked_token
    trimmed_string = cleaned_string.strip()

        # The synthesizer works in batch, so you need to put your data in a list or numpy array
    text_arr = []
    if len(trimmed_string) != 0:
      text_arr = [trimmed_string]
    if text_arr:
        print(text_arr)
        embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = synthesizer.synthesize_spectrograms(text_arr, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")


        ## Generating the waveform
        print("Synthesizing the waveform:")
        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)


        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Save it on the disk
        fpath = "output/%02d - %s.wav" % (num_generated, trimmed_string[:10])
        print(generated_wav)
        librosa.output.write_wav(fpath, generated_wav, synthesizer.sample_rate)
        num_generated += 1
        print("Saved output as %s\n\n" % fpath)
        outfiles.append(fpath)

# Concatenate the outfiles and stretch to certain bpm

i = 0
desired_tempo = 100
outfpath = "output/%s-%sbpm.flac" % (list(document.sents)[0].string[:10], desired_tempo)

for fpath in list(outfiles):

      # Load the file from disk, trim excess silence, slow it down a bit and
      # concatenate it to the full file

      sentence, sr = librosa.core.load(fpath)
      trimmed, index = librosa.effects.trim(sentence)

      onset_env = librosa.onset.onset_strength(y=trimmed, sr=sr)
      tempo = librosa.beat.tempo(y=trimmed, onset_envelope=onset_env)
      rate = desired_tempo / int(tempo)

      print(fpath, tempo, rate)
      slowed = pyrb.time_stretch(trimmed, sr, rate)

      if i != 0:
          outf, sr = librosa.core.load(outfpath)
          z = np.append(outf, slowed)
          sf.write(outfpath, z, sr, format='flac', subtype='PCM_24')
      else:
          sf.write(outfpath, slowed, sr, format='flac', subtype='PCM_24')

      i += 1

print("\nSaved whole text as %s\n\
