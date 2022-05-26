import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import pickle


print("TensorFlow version:", tf.__version__)


FILE_PATH = "./UrbanSound8K/audio/fold"

dict_training = (1,2,3,4,6)


audios = pd.read_csv("./UrbanSound8K/metadata/UrbanSound8K.csv")
audios.head()


SAMPLING_RATE = 44100
NUM_SAMPLES = 44100*4
NMFCC = 15
NFFT = 20148
HOP_LEN = 512
NMEL = 40


first_train = first_test5 = first_test7 = first_test8 = first_test9 = first_test10 = True

for index, row in audios.iterrows(): #Reading the csv file with metadata row by row
    
    folder = row.fold
    
    audio_name = FILE_PATH+str(folder)+"/"+row.slice_file_name
    
    sound_file, sampling_rate = librosa.load(audio_name, sr = None, mono=False) 
    #print(sound_file.shape)

    #Data pre-processing
    if sampling_rate != SAMPLING_RATE:
        sound_file = librosa.resample(y=sound_file, target_sr=SAMPLING_RATE, orig_sr=sampling_rate)

    if sound_file.shape[0] == 2:
        sound_file = np.mean(sound_file, axis=0)
    #print(sound_file.shape)

    if sound_file.shape[0] != NUM_SAMPLES:
        if sound_file.shape[0] > NUM_SAMPLES:
            sound_file = sound_file[:NUM_SAMPLES]
        else:
            sound_file = np.pad(sound_file,(0,NUM_SAMPLES-sound_file.shape[0]))

    #Feature extraction
    mfccs = librosa.feature.mfcc(y=sound_file, n_mfcc = NMFCC, sr=SAMPLING_RATE)
    #first and sec der of mfccs = delta and delta delta mfccs
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order = 2)

    mfccs_tot = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
 
    mel_spectogram = librosa.feature.melspectrogram(y=sound_file, sr=SAMPLING_RATE, n_fft=NFFT, hop_length=HOP_LEN, n_mels=NMEL)
    
    chroma = librosa.feature.chroma_stft(y=sound_file, sr=SAMPLING_RATE, n_fft=NFFT, hop_length=HOP_LEN, n_chroma=12)
   
    rms_sig = librosa.feature.rms(y=sound_file, frame_length=NFFT, hop_length=HOP_LEN)
    
    rms_stat = [np.min(rms_sig),np.max(rms_sig),np.mean(rms_sig), np.std(rms_sig)]    
    
    #Splitting into train and test set
    if folder in dict_training:
        if first_train:
            train_labels = np.array(row['classID'])
            train_mfccs = np.array([mfccs_tot])
            train_melSpect = np.array([mel_spectogram])
            train_chroma = np.array([chroma])
            train_rms_stat = np.array([rms_stat])
            first_train = False
        else:
            train_labels = np.append(train_labels, row['classID'])
            train_mfccs = np.concatenate((train_mfccs, np.array([mfccs_tot])))
            train_melSpect = np.concatenate((train_melSpect, np.array([mel_spectogram])))
            train_chroma = np.concatenate((train_chroma, np.array([chroma])))
            train_rms_stat = np.concatenate((train_rms_stat, np.array([rms_stat])))
            
    elif folder == 5:
        if first_test5:
            test5_labels = np.array(row['classID'])
            test5_mfccs = np.array([mfccs_tot])
            test5_melSpect = np.array([mel_spectogram])
            test5_chroma = np.array([chroma])
            test5_rms_stat = np.array([rms_stat])
            first_test5 = False
        else:
            #test5_labels = np.concatenate((test5_labels, np.array([row['classID']])))
            test5_labels=np.append(test5_labels,row['classID'])
            test5_mfccs = np.concatenate((test5_mfccs, np.array([mfccs_tot])))
            test5_melSpect = np.concatenate((test5_melSpect, np.array([mel_spectogram])))
            test5_chroma = np.concatenate((test5_chroma, np.array([chroma])))
            test5_rms_stat = np.concatenate((test5_rms_stat, np.array([rms_stat])))

    elif folder == 7:
        if first_test7:
            test7_labels = np.array([row['classID']])
            test7_mfccs = np.array([mfccs_tot])
            test7_melSpect = np.array([mel_spectogram])
            test7_chroma = np.array([chroma])
            test7_rms_stat = np.array([rms_stat])
            first_test7 = False
        else:
            test7_labels = np.append(test7_labels, row['classID'])
            test7_mfccs = np.concatenate((test7_mfccs, np.array([mfccs_tot])))
            test7_melSpect = np.concatenate((test7_melSpect, np.array([mel_spectogram])))
            test7_chroma = np.concatenate((test7_chroma, np.array([chroma])))
            test7_rms_stat = np.concatenate((test7_rms_stat, np.array([rms_stat])))

    elif folder == 8:
        if first_test8:
            test8_labels = np.array([row['classID']])
            test8_mfccs = np.array([mfccs_tot])
            test8_melSpect = np.array([mel_spectogram])
            test8_chroma = np.array([chroma])
            test8_rms_stat = np.array([rms_stat])
            first_test8 = False
        else:
            test8_labels = np.append(test8_labels, row['classID'])
            test8_mfccs = np.concatenate((test8_mfccs, np.array([mfccs_tot])))
            test8_melSpect = np.concatenate((test8_melSpect, np.array([mel_spectogram])))
            test8_chroma = np.concatenate((test8_chroma, np.array([chroma])))
            test8_rms_stat = np.concatenate((test8_rms_stat, np.array([rms_stat])))

    elif folder == 9:
        if first_test9:
            test9_labels = np.array([row['classID']])
            test9_mfccs = np.array([mfccs_tot])
            test9_melSpect = np.array([mel_spectogram])
            test9_chroma = np.array([chroma])
            test9_rms_stat = np.array([rms_stat])
            first_test9 = False
        else:
            test9_labels = np.append(test9_labels, row['classID'])
            test9_mfccs = np.concatenate((test9_mfccs, np.array([mfccs_tot])))
            test9_melSpect = np.concatenate((test9_melSpect, np.array([mel_spectogram])))
            test9_chroma = np.concatenate((test9_chroma, np.array([chroma])))
            test9_rms_stat = np.concatenate((test9_rms_stat, np.array([rms_stat])))

    elif folder == 10:
        if first_test10:
            test10_labels = np.array([row['classID']])
            test10_mfccs = np.array([mfccs_tot])
            test10_melSpect = np.array([mel_spectogram])
            test10_chroma = np.array([chroma])
            test10_rms_stat = np.array([rms_stat])
            first_test10 = False
        else:
            test10_labels = np.append(test10_labels, row['classID'])
            test10_mfccs = np.concatenate((test10_mfccs, np.array([mfccs_tot])))
            test10_melSpect = np.concatenate((test10_melSpect, np.array([mel_spectogram])))
            test10_chroma = np.concatenate((test10_chroma, np.array([chroma])))
            test10_rms_stat = np.concatenate((test10_rms_stat, np.array([rms_stat])))


print(train_rms_stat.shape)

train_labels = train_labels.reshape(train_labels.shape[0],1)
stat_df_train = np.concatenate((train_rms_stat,np.array(train_labels)),axis=1)

# Creating a csv file with RMS stats
#stat_df_train = pd.DataFrame([train_rms_stat[:,0],train_rms_stat[:,1],train_rms_stat[:,2],train_rms_stat[:,3],train_labels],columns=["min","max","mean","std","label"])
stat_df_train = pd.DataFrame(stat_df_train,columns=["min","max","mean","std","label"])
stat_df_train.to_csv("stat_train.csv")

test5_labels = test5_labels.reshape(test5_labels.shape[0],1)
stat_df_test5 = np.concatenate((test5_rms_stat,np.array(test5_labels)),axis=1)
stat_df_test5 = pd.DataFrame(stat_df_test5,columns=["min","max","mean","std","label"])

test7_labels = test7_labels.reshape(test7_labels.shape[0],1)
stat_df_test7 = np.concatenate((test7_rms_stat,np.array(test7_labels)),axis=1)
stat_df_test7 = pd.DataFrame(stat_df_test7,columns=["min","max","mean","std","label"])

test8_labels = test8_labels.reshape(test8_labels.shape[0],1)
stat_df_test8 = np.concatenate((test8_rms_stat,np.array(test8_labels)),axis=1)
stat_df_test8 = pd.DataFrame(stat_df_test8,columns=["min","max","mean","std","label"])

test9_labels = test9_labels.reshape(test9_labels.shape[0],1)
stat_df_test9 = np.concatenate((test9_rms_stat,np.array(test9_labels)),axis=1)
stat_df_test9 = pd.DataFrame(stat_df_test9,columns=["min","max","mean","std","label"])

test10_labels = test10_labels.reshape(test10_labels.shape[0],1)
stat_df_test10 = np.concatenate((test10_rms_stat,np.array(test10_labels)),axis=1)
stat_df_test10 = pd.DataFrame(stat_df_test10,columns=["min","max","mean","std","label"])

stat_df_test = stat_df_test5.append(stat_df_test7,ignore_index=True)
stat_df_test = stat_df_test.append(stat_df_test8,ignore_index=True)
stat_df_test = stat_df_test.append(stat_df_test9,ignore_index=True)
stat_df_test = stat_df_test.append(stat_df_test10,ignore_index=True)

stat_df_test.to_csv("stat_test.csv")


#Saving pickle objects split into train and test
names = {"mfcc_train":train_mfccs,"labels_train":train_labels,"mfcc_test5":test5_mfccs,"labels_test5":test5_labels,"mfcc_test7":test7_mfccs,
        "labels_test7":test7_labels,"mfcc_test8":test8_mfccs,"labels_test8":test8_labels,"mfcc_test9":test9_mfccs,"labels_test9":test9_labels,
        "mfcc_test10":test10_mfccs,"labels_test10":test10_labels,
        "melSpect_train":train_melSpect,"melSpect_test5":test5_melSpect,"melSpect_test7":test7_melSpect,
        "melSpect_test8":test8_melSpect,"melSpect_test9":test9_melSpect,"melSpect_test10":test10_melSpect,
        "chroma_train":train_chroma,"chroma_test5":test5_chroma,"chroma_test7":test7_chroma,
        "chroma_test8":test8_chroma,"chroma_test9":test9_chroma,"chroma_test10":test10_chroma,
        "rmsStat_train":train_rms_stat,"rmsStat_test5":test5_rms_stat,"rmsStat_test7":test7_rms_stat,
        "rmsStat_test8":test8_rms_stat,"rmsStat_test9":test9_rms_stat,"rmsStat_test10":test10_rms_stat}


for key in names:
    open_file = open(key+".pkl", "wb")
    pickle.dump(names[key], open_file)
    open_file.close()
