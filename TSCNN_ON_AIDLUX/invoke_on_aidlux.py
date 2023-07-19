import onnxruntime
import numpy as np
import librosa
import numpy as np
import warnings
import time
import android
import datetime
droid = android.Android()
warnings.filterwarnings("ignore")
#MLMC预处理
def get_features(y, sr):
    hop_l = 512*2
    f1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60, hop_length=hop_l)
    f2 = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_l)
    f3 = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_l)
    f4 = librosa.effects.harmonic(y)
    f4 = librosa.feature.tonnetz(y=f4, sr=sr, hop_length=hop_l)
    f5 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=60, hop_length=hop_l)
    f5 = librosa.power_to_db(f5)
    MLMC = np.vstack((f1, f5, f2, f3, f4))
    out = MLMC.reshape((1, 1, 145, 55))
    out = out.astype(np.float32)
    return out
def get_features_file(path):
    y, sr=librosa.load(path)
    hop_l = 512*2
    f1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60, hop_length=hop_l)
    f2 = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_l)
    f3 = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_l)
    f4 = librosa.effects.harmonic(y)
    f4 = librosa.feature.tonnetz(y=f4, sr=sr, hop_length=hop_l)
    f5 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=60, hop_length=hop_l)
    f5 = librosa.power_to_db(f5)
    MLMC = np.vstack((f1, f5, f2, f3, f4))
    out = MLMC.reshape((1, 1, 145, 55))
    out = out.astype(np.float32)
    return out
while True:
    droid.recorderStartMicrophone('/sdcard/Music/test.amr')
    time.sleep(2.53)
    droid.recorderStop()
    input_arr,Mel = get_features_file('/sdcard/Music/test.amr')
    model = onnxruntime.InferenceSession('tscnn.onnx')
    inputs = {'input':input_arr}
    outputs_n = model.run(['266'],inputs)[0]
    pred_value, pred_index = np.max(outputs_n),np.argmax(outputs_n)
    cls = ['calm', 'feeding', 'frightened', 'anxious']
    pred_emo = cls[pred_index]
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),f'predict emotion: [{pred_emo}], possibility: [{pred_value}]')

