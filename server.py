from flask import Flask, redirect, request, jsonify, render_template
import soundfile as sf
import librosa
import numpy as np
from os import system
import speechmetrics

import math

app = Flask(__name__, static_url_path='/', static_folder='static')

texts = ['Alla tiger och undviker varandra med blicken',
        'De kanske annars skulle ha flugit söderut mot värmen',
        'Fotboll intresserade honom ju egentligen inte',
        'Ingen av de tillfrågade ville syssla med gruvdrift',
        'Vid juletid skickas mössor till kungafamiljen',
        'Avhoppet från överklassen innebar ett vägval',
        'Bara ett syskon kunde tänkas ha samma blodgrupp',
        'Den ljushårige  publikfavoriten skulle bort',
        'Han skulle gifta sig med en ängel utan vingar',
        'Han rullade sina axlar för att mjuka upp lederna']

speech_metrics = speechmetrics.load('absolute', window=None)

@app.route('/')
@app.route('/app/'+str(len(texts)+1))# Loop if all are done
def redir(index=0):
  return redirect('/app/0', code=302)


@app.route('/app/<index>')
def display_page(index=0):
  
  return render_template('index.html', index=int(index), text=texts[int(index)])
  

@app.route('/thing', methods=['POST'])
def thing():
  f = request.files['file']
  f.save(f.filename) # Saves the file
  wav_filename = 'tmp.wav'
  # Convert to wav (file is in ogg format, so thing don't work with sf or sr)
  system('ffmpeg -y -i {} -ac 1 {}'.format(f.filename, wav_filename))

  metrics = get_metrics(wav_filename)
  metrics['pitch_score'] = (metrics['stddev'] - 4.552686) / 1.998870
  metrics['voice_score'] = (metrics['voiced_prob'] - 0.461134) / 0.140464 + 0.3*(4.012101 - metrics['mosnet']) /  0.355457
  return jsonify(metrics)

def get_metrics(wav_filename='tmp.wav'):
  speech_mets = speech_metrics(wav_filename)
  data, sr = librosa.load(wav_filename, sr=16000)
  spectrogram = librosa.feature.melspectrogram(y=data, sr=sr)
  f0, _, prob = librosa.pyin(data, sr=sr, fmin=65, fmax=600, frame_length=512)
  # Filter out noise in f0
  prob[f0==65] = np.nan
  f0[f0==65] = np.nan
  syllables = count_syllables(f0)

  f0, _, prob = librosa.pyin(data, sr=sr, fmin=65, fmax=600)
  # Filter out noise in f0
  prob[f0==65] = np.nan
  f0[f0==65] = np.nan

  prob[prob <= 0.02] = np.nan
  f0[prob <= 0.02] = np.nan

  # Trim silence
  f0, prob = trim_silence(f0, prob)
  if f0.size < 50:#Ignore really short sequences
    return {}
  f0 = librosa.hz_to_midi(f0)
  return {'voiced_prob': np.nanmean(np.where(np.isnan(f0), f0, prob)),
                  "f0": np.nanmean(f0),
                  "stddev": np.nanstd(f0),
                  'syllable_estimate': syllables,
                  'voiced_percent': float(np.count_nonzero(np.where(np.isnan(f0), 0, 1)))/f0.size,
                  'srmr': speech_mets['srmr'],
                  'mosnet': float(speech_mets['mosnet'][0][0]),
                  'length': len(f0),
                  'filename': wav_filename}



def trim_silence(f0, prob):
  i = 0
  for f in f0:
    if np.isnan(f):
      i += 1
    else:
      break

  f0 = f0[i:]
  prob = prob[i:]
  i = 0
  for f in reversed(f0):
    if np.isnan(f):
      i += 1
    else:
      break
  
  f0 = f0[:-i]
  prob = prob[:-i]
  return f0, prob


def count_syllables(f0):
  frame_length = 512.0/16000.0 #Frame length in seconds
  #Maximum pause length in frames (pause length in seconds is 0.3)
  pause_length = int(0.8 / frame_length)
  max_consecutive_syl = 0
  n = 0
  breaks = 0
  br  = False
  for i in f0:
    if math.isnan(i):
      if br:
        n += 1
      br = False
      breaks += 1
      if breaks >= pause_length:
        max_consecutive_syl = max(max_consecutive_syl, n)
        n = 0
    else:
      br = True
      breaks = 0

  max_consecutive_syl = max(max_consecutive_syl, n)
  return max_consecutive_syl


if __name__ == '__main__':
  app.run()
