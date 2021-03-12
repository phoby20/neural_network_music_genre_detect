from django.shortcuts import render
from .forms import *
from .neuralnetwork import *
import librosa

import numpy as np
import pandas as pd
import scipy.special
import pickle
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import json
import os.path

# Create your views here.

def homeView(request):
    params = {
        'music_forms':music_form()
    }
    # print(params['music_forms'])

    if (request.method == 'POST'):
        input_nodes = 57
        hidden_nodes = 500
        output_nodes = 10
        learning_rate = 0.00005
        n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

        dic = {'reggae': [0], 'country': [1], 'jazz': [2], 'hiphop': [3], 'metal': [4], 'disco': [5], 'classical': [6],
               'rock': [7], 'blues': [8], 'pop': [9]}

        sound = request.FILES.get('sound_file')
        y, sr = librosa.load(sound)
        chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=512)
        spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        y_harm, y_perc = librosa.effects.hpss(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        chroma_stft_mean = chromagram.mean()+0.0
        chroma_stft_var = chromagram.var()+0.0
        spectral_centroids_mean = spectral_centroids.mean()+0.0
        spectral_centroids_var = spectral_centroids.var()+0.0
        spectral_bandwidth_mean = spectral_bandwidth.mean()+0.0
        spectral_bandwidth_var = spectral_bandwidth.var()+0.0
        rms_mean = rms.mean()+0.0
        rms_var = rms.var()+0.0
        rolloff_mean = rolloff.mean()+0.0
        rolloff_var = rolloff.var()+0.0
        zero_crossing_rate_mean = zero_crossing_rate.mean()+0.0
        zero_crossing_rate_var = zero_crossing_rate.var()+0.0
        harm_mean = y_harm.mean()+0.0
        harm_var = y_harm.var()+0.0
        tempo, _ = librosa.beat.beat_track(y, sr=sr)
        y_perc_mean = y_perc.mean()+0.0
        y_perc_var = y_perc.var()+0.0

        mfcc1_mean = mfcc[0].mean()
        mfcc1_var = mfcc[0].var()
        mfcc2_mean = mfcc[1].mean()
        mfcc2_var = mfcc[1].var()
        mfcc3_mean = mfcc[2].mean()
        mfcc3_var = mfcc[2].var()
        mfcc4_mean = mfcc[3].mean()
        mfcc4_var = mfcc[3].var()
        mfcc5_mean = mfcc[4].mean()
        mfcc5_var = mfcc[4].var()

        mfcc6_mean = mfcc[5].mean()
        mfcc6_var = mfcc[5].var()
        mfcc7_mean = mfcc[6].mean()
        mfcc7_var = mfcc[6].var()
        mfcc8_mean = mfcc[7].mean()
        mfcc8_var = mfcc[7].var()
        mfcc9_mean = mfcc[8].mean()
        mfcc9_var = mfcc[8].var()
        mfcc10_mean = mfcc[9].mean()
        mfcc10_var = mfcc[9].var()

        mfcc11_mean = mfcc[10].mean()
        mfcc11_var = mfcc[10].var()
        mfcc12_mean = mfcc[11].mean()
        mfcc12_var = mfcc[11].var()
        mfcc13_mean = mfcc[12].mean()
        mfcc13_var = mfcc[12].var()
        mfcc14_mean = mfcc[13].mean()
        mfcc14_var = mfcc[13].var()
        mfcc15_mean = mfcc[14].mean()
        mfcc15_var = mfcc[14].var()

        mfcc16_mean = mfcc[15].mean()
        mfcc16_var = mfcc[15].var()
        mfcc17_mean = mfcc[16].mean()
        mfcc17_var = mfcc[16].var()
        mfcc18_mean = mfcc[17].mean()
        mfcc18_var = mfcc[17].var()
        mfcc19_mean = mfcc[18].mean()
        mfcc19_var = mfcc[18].var()
        mfcc20_mean = mfcc[19].mean()
        mfcc20_var = mfcc[19].var()

        anal_list = [chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroids_mean, spectral_centroids_var,
                     spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean, rolloff_var, zero_crossing_rate_mean,
                     zero_crossing_rate_var, harm_mean, harm_var, y_perc_mean, y_perc_var, tempo]

        for i in range(len(mfcc)):
            anal_list.append(mfcc[i].mean())
            anal_list.append(mfcc[i].var())

        # df = pd.DataFrame(anal_list).T
        # df.columns = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 'spectral_centroids_mean', 'spectral_centroids_var',
        #               'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean',
        #               'zero_crossing_rate_var', 'harm_mean', 'harm_var', 'y_perc_mean', 'y_perc_var', 'tempo',
        #               'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var ','mfcc4_mean',
        #               'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var',
        #               'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean',
        #               'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var',
        #               'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
        #               'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']

        d = {'chroma_stft_mean': [chroma_stft_mean], 'chroma_stft_var': [chroma_stft_var], 'rms_mean': [rms_mean],
             'rms_var': [rms_var], 'spectral_centroids_mean': [spectral_centroids_mean], 'spectral_centroids_var': [spectral_centroids_var],
             'spectral_bandwidth_mean': [spectral_bandwidth_mean], 'spectral_bandwidth_var': [spectral_bandwidth_var], 'rolloff_mean': [rolloff_mean],
             'rolloff_var': [rolloff_var], 'zero_crossing_rate_mean': [zero_crossing_rate_mean], 'zero_crossing_rate_var': [zero_crossing_rate_var],
             'harm_mean': [harm_mean], 'harm_var': [harm_var], 'y_perc_mean': [y_perc_mean], 'y_perc_var': [y_perc_var],
             'tempo': [tempo], 'mfcc1_mean': [mfcc1_mean], 'mfcc1_var': [mfcc1_var], 'mfcc2_mean': [mfcc2_mean],
             'mfcc2_var': [mfcc2_var],
             'mfcc3_mean': [mfcc3_mean], 'mfcc3_var': [mfcc3_var], 'mfcc4_mean': [mfcc4_mean],
             'mfcc4_var': [mfcc4_var], 'mfcc5_mean': [mfcc5_mean], 'mfcc5_var': [mfcc5_var],
             'mfcc6_mean': [mfcc6_mean], 'mfcc6_var': [mfcc6_var], 'mfcc7_mean': [mfcc7_mean],
             'mfcc7_var': [mfcc7_var], 'mfcc8_mean': [mfcc8_mean], 'mfcc8_var': [mfcc8_var],
             'mfcc9_mean': [mfcc9_mean], 'mfcc9_var': [mfcc9_var], 'mfcc10_mean': [mfcc10_mean],
             'mfcc10_var': [mfcc10_var], 'mfcc11_mean': [mfcc11_mean], 'mfcc11_var': [mfcc11_var],
             'mfcc12_mean': [mfcc12_mean], 'mfcc12_var': [mfcc12_var], 'mfcc13_mean': [mfcc13_mean],
             'mfcc13_var': [mfcc13_var], 'mfcc14_mean': [mfcc14_mean], 'mfcc14_var': [mfcc14_var],
             'mfcc15_mean': [mfcc15_mean], 'mfcc15_var': [mfcc15_var], 'mfcc16_mean': [mfcc16_mean],
             'mfcc16_var': [mfcc16_var], 'mfcc17_mean': [mfcc17_mean], 'mfcc17_var': [mfcc17_var],
             'mfcc18_mean': [mfcc18_mean], 'mfcc18_var': [mfcc18_var], 'mfcc19_mean': [mfcc19_mean],
             'mfcc19_var': [mfcc19_var], 'mfcc20_mean': [mfcc20_mean], 'mfcc20_var': [mfcc20_var],
             }
        df = pd.DataFrame(data=d)


        # 정규화가 안됨........................................................2021-03-12
        scaler = sklearn.preprocessing.MinMaxScaler()
        np_scaled = scaler.fit_transform(anal_list)
        # X = pd.DataFrame(np_scaled, columns=df.columns)


        for e in range(len(df)):
            # targets = numpy.zeros(output_nodes) + 0.01
            # dl = y_test[e:e + 1].to_numpy()
            # targets_test[int(dic[dl[0]][0])] = 0.99
            # print('지금까지의 진행도 : ', count)
            # count += 1
            output = n.query(np_scaled)
            label = np.argmax(output)
            # query_result.append(label)



        print()
        # print('np_scaled: ', np_scaled)
        # print(label)
        # for i in df:
        #
        #     print('i:', df[i])
        print('np_scaled', np_scaled)

        return render(request, 'index/index.html', {'params':params})
    return render(request, 'index/index.html', {'params':params})