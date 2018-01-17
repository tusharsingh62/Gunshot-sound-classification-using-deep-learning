
import librosa
import numpy as np
import librosa.display
from librosa.feature import melspectrogram
import matplotlib.pyplot as plt

y, sr = librosa.load('/home/administrator/Downloads/ML_work/Gunshot_detection_CNN/shotgun.mp3')

melspectrogram(y=y, sr=sr)

D = np.abs(librosa.stft(y))**2

S = librosa.feature.melspectrogram(S=D)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax = 8000)

plt.figure(figsize=(12, 4))

data = librosa.power_to_db(S, ref = np.max)

librosa.display.specshow(data,y_axis = 'mel', x_axis = 'time', fmax = 8000)                                          
                           
                          
plt.colorbar(format='%+2.0f dB')
plt.title('Mel scaled spectrogram')
plt.tight_layout()
plt.show()
