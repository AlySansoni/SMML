import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import pandas as pd
import librosa

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

PCA_COMPONENT = 50
print("TensorFlow version:", tf.__version__)

N_CHANNELS = 3
BATCH_SIZE = 128

def plt_history(history, n_epochs):

    print(n_epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(n_epochs+1)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    return plt.show()


def plt_confusion_matrix(cm):

# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(cm,
                        index = ['0','1','2','3','4','5','6','7','8','9'], 
                        columns = ['0','1','2','3','4','5','6','7','8','9'])
    #Plotting the confusion matrix in terms of percentages
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df/np.sum(cm_df), annot=True, cmap='Blues', fmt='.2%')
    #sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
    #        fmt='.2%', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()

#Plot of the feature distribution in the space
def plt_pca(data, label, title):
    #print(data.shape)
    cm = plt.cm.get_cmap('RdYlBu')
    data_2d = PCA(n_components=2).fit_transform(data.reshape(len(data),-1))
    plt.scatter(data_2d[:,0],data_2d[:,1],c=label,cmap=cm)
    plt.title(title)
    plt.colorbar()
    #plt.show()
    title = title.replace(' ', '_')
    plt.savefig("Images/"+title+'.png')
    plt.clf()
    
    #pd.DataFrame(data[:15].reshape(15,-1)).T.plot()


#Function that tests the trained model
def test_model(model, test5, test7, test8, test9, test10, y_5, y_7, y_8, y_9, y_10, batch_size):
    print("Testing model...")
    cvscores = []

    print(test5.shape)
    predictions5 = [0]*len(test5)
    results5 = model.evaluate(test5, y_5, batch_size = batch_size)
    for i in range(len(test5)):
        if N_CHANNELS == 1:
            predictions5[i]=model.predict(test5[i].reshape(1, test5[i].shape[0],test5[i].shape[1])).argmax(axis=1)
        else:
            predictions5[i]=model.predict(test5[i].reshape(1, test5[i].shape[0],test5[i].shape[1],test5[i].shape[2])).argmax(axis=1)
    print("Folder 5 test loss: ",results5[0], " test acc: ", results5[1])
    print("Folder 5 confusion matrix:\n", confusion_matrix(y_5[:,0], predictions5))
    cvscores.append(results5[1]*100)

    predictions7 = [0]*len(test7)
    results7 = model.evaluate(test7, y_7, batch_size = batch_size)
    for i in range(len(test7)):
        if N_CHANNELS == 1:
            predictions7[i]=model.predict(test7[i].reshape(1, test7[i].shape[0],test7[i].shape[1])).argmax(axis=1)
        else:
            predictions7[i]=model.predict(test7[i].reshape(1, test7[i].shape[0],test7[i].shape[1],test7[i].shape[2])).argmax(axis=1)
    print("Folder 7 test loss: ",results7[0], " test acc: ", results7[1])
    print("Folder 7 confusion matrix:\n", confusion_matrix(y_7, predictions7))
    cvscores.append(results7[1]*100)

    predictions8 = [0]*len(test8)
    results8 = model.evaluate(test8, y_8, batch_size = batch_size)
    for i in range(len(test8)):
        if N_CHANNELS == 1:
            predictions8[i]=model.predict(test8[i].reshape(1, test8[i].shape[0],test8[i].shape[1])).argmax(axis=1)
        else:
            predictions8[i]=model.predict(test8[i].reshape(1, test8[i].shape[0],test8[i].shape[1],test8[i].shape[2])).argmax(axis=1)
    print("Folder 8 test loss: ",results8[0], " test acc: ", results8[1])
    print("Folder 8 confusion matrix:\n", confusion_matrix(y_8, predictions8))
    cvscores.append(results8[1]*100)

    predictions9 = [0]*len(test9)
    results9 = model.evaluate(test9, y_9, batch_size = batch_size)
    for i in range(len(test9)):
        if N_CHANNELS == 1:
            predictions9[i]=model.predict(test9[i].reshape(1, test9[i].shape[0],test9[i].shape[1])).argmax(axis=1)
        else:
            predictions9[i]=model.predict(test9[i].reshape(1, test9[i].shape[0],test9[i].shape[1],test9[i].shape[2])).argmax(axis=1)
    print("Folder 9 test loss: ",results9[0], " test acc: ", results9[1])
    print("Folder 9 confusion matrix:\n", confusion_matrix(y_9, predictions9))
    cvscores.append(results9[1]*100)

    predictions10 = [0]*len(test10)
    results10 = model.evaluate(test10, y_10, batch_size = batch_size)
    for i in range(len(test10)):
        if N_CHANNELS == 1:
            predictions10[i]=model.predict(test10[i].reshape(1, test10[i].shape[0],test10[i].shape[1])).argmax(axis=1)
        else:
            predictions10[i]=model.predict(test10[i].reshape(1, test10[i].shape[0],test10[i].shape[1],test10[i].shape[2])).argmax(axis=1)
    print("Folder 10 test loss: ",results10[0], " test acc: ", results10[1])
    print("Folder 10 confusion matrix:\n", confusion_matrix(y_10, predictions10))
    cvscores.append(results10[1]*100)

    y_true = np.concatenate((y_5, y_7, y_8, y_9, y_10))
    y_pred = np.concatenate((predictions5, predictions7, predictions8, predictions9, predictions10))

    print("Average test accuracy and std: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    cm = confusion_matrix(y_true, y_pred)
    print("Global Confusion Matrix:\n", cm)
    plt_confusion_matrix(cm)


#Once the model is designed, here it can be trained
def selected_model(model, Xtrain, Ytrain, lr, batch_size, n_epochs, sel_test5, sel_test7, sel_test8, sel_test9, sel_test10, test_labels5,test_labels7, test_labels8, test_labels9, test_labels10):

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                        optimizer = tf.optimizers.Adam(learning_rate = lr), metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

    history = model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=n_epochs, verbose='auto', callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.1, shuffle=True)
    
    model.load_weights(filepath = '.mdl_wts.hdf5')
    #Train and Test The Model

    test_model(model, sel_test5, sel_test7, sel_test8, sel_test9, sel_test10, test_labels5, test_labels7, 
                test_labels8, test_labels9, test_labels10, batch_size)

    plt_history(history, earlyStopping.stopped_epoch)


  
#Loading all labels

train_labels = pickle.load(open('Train/labels_train.pkl', 'rb'))
test_labels5 = pickle.load(open('Test/labels_test5.pkl', 'rb'))
test_labels7 = pickle.load(open('Test/labels_test7.pkl', 'rb'))
test_labels8 = pickle.load(open('Test/labels_test8.pkl', 'rb'))
test_labels9 = pickle.load(open('Test/labels_test9.pkl', 'rb'))
test_labels10 = pickle.load(open('Test/labels_test10.pkl', 'rb'))


#Loading mfccs

train_mfccs = pickle.load(open('Train/mfcc_train.pkl', "rb"))
test_mfccs5 = pickle.load(open('Test/mfcc_test5.pkl', "rb"))
test_mfccs7 = pickle.load(open('Test/mfcc_test7.pkl', "rb"))
test_mfccs8 = pickle.load(open('Test/mfcc_test8.pkl', "rb"))
test_mfccs9 = pickle.load(open('Test/mfcc_test9.pkl', "rb"))
test_mfccs10 = pickle.load(open('Test/mfcc_test10.pkl', "rb"))

#Normalizing mfccs
scaler = StandardScaler().fit(train_mfccs.reshape(-1, train_mfccs.shape[-1]))
train_mfccs = scaler.transform(train_mfccs.reshape(-1, train_mfccs.shape[-1])).reshape(train_mfccs.shape)
test_mfccs5 = scaler.transform(test_mfccs5.reshape(-1, test_mfccs5.shape[-1])).reshape(test_mfccs5.shape)
test_mfccs7 = scaler.transform(test_mfccs7.reshape(-1, test_mfccs7.shape[-1])).reshape(test_mfccs7.shape)
test_mfccs8 = scaler.transform(test_mfccs8.reshape(-1, test_mfccs8.shape[-1])).reshape(test_mfccs8.shape)
test_mfccs9 = scaler.transform(test_mfccs9.reshape(-1, test_mfccs9.shape[-1])).reshape(test_mfccs9.shape)
test_mfccs10 = scaler.transform(test_mfccs10.reshape(-1, test_mfccs10.shape[-1])).reshape(test_mfccs10.shape)

"""#Printing the location in space
plt_pca(train_mfccs, train_labels, "Training MFCC")
plt_pca(test_mfccs5, test_labels5, "Test 5 MFCC")
plt_pca(test_mfccs7, test_labels7, "Test 7 MFCC")
plt_pca(test_mfccs8, test_labels8, "Test 8 MFCC")
plt_pca(test_mfccs9, test_labels9, "Test 9 MFCC")
plt_pca(test_mfccs10, test_labels10, "Test 10 MFCC")"""

#Loading mfcc with 25 instead of 15 coefficiets 
train_mfccs25 = pickle.load(open('Train/mfcc_25train.pkl', "rb"))
test5_mfccs25 = pickle.load(open('Test/mfcc_25test5.pkl', "rb"))
test7_mfccs25 = pickle.load(open('Test/mfcc_25test7.pkl', "rb"))
test8_mfccs25 = pickle.load(open('Test/mfcc_25test8.pkl', "rb"))
test9_mfccs25 = pickle.load(open('Test/mfcc_25test9.pkl', "rb"))
test10_mfccs25 = pickle.load(open('Test/mfcc_25test10.pkl', "rb"))

#Normalizing mfccs with 25 mfccs
scaler = StandardScaler().fit(train_mfccs25.reshape(-1, train_mfccs25.shape[-1]))
train_mfccs25 = scaler.transform(train_mfccs25.reshape(-1, train_mfccs25.shape[-1])).reshape(train_mfccs25.shape)
test5_mfccs25 = scaler.transform(test5_mfccs25.reshape(-1, test5_mfccs25.shape[-1])).reshape(test5_mfccs25.shape)
test7_mfccs25 = scaler.transform(test7_mfccs25.reshape(-1, test7_mfccs25.shape[-1])).reshape(test7_mfccs25.shape)
test8_mfccs25 = scaler.transform(test8_mfccs25.reshape(-1, test8_mfccs25.shape[-1])).reshape(test8_mfccs25.shape)
test9_mfccs25 = scaler.transform(test9_mfccs25.reshape(-1, test9_mfccs25.shape[-1])).reshape(test9_mfccs25.shape)
test10_mfccs25 = scaler.transform(test10_mfccs25.reshape(-1, test10_mfccs25.shape[-1])).reshape(test10_mfccs25.shape)

#print(train_mfccs25.shape)

"""plt_pca(train_mfccs25, train_labels, "Training MFCC 25")
plt_pca(test5_mfccs25, test_labels5, "Test 5 MFCC 25")
plt_pca(test7_mfccs25, test_labels7, "Test 7 MFCC 25")
plt_pca(test8_mfccs25, test_labels8, "Test 8 MFCC 25")
plt_pca(test9_mfccs25, test_labels9, "Test 9 MFCC 25")
plt_pca(test10_mfccs25, test_labels10, "Test 10 MFCC 25")"""

#Loading melSpectograms

train_melSpect = pickle.load(open('Train/melSpect_train.pkl', "rb"))
test_melSpect5 = pickle.load(open('Test/melSpect_test5.pkl', "rb"))
test_melSpect7 = pickle.load(open('Test/melSpect_test7.pkl', "rb"))
test_melSpect8 = pickle.load(open('Test/melSpect_test8.pkl', "rb"))
test_melSpect9 = pickle.load(open('Test/melSpect_test9.pkl', "rb"))
test_melSpect10 = pickle.load(open('Test/melSpect_test10.pkl', "rb"))

#Normalizing melSpectogram
for i in range(len(train_melSpect)):
    train_melSpect[i] = librosa.util.normalize(train_melSpect[i])
for i in range(len(test_melSpect5)):
    test_melSpect5[i] = librosa.util.normalize(test_melSpect5[i])
for i in range(len(test_melSpect7)):
    test_melSpect7[i] = librosa.util.normalize(test_melSpect7[i])
for i in range(len(test_melSpect8)):
    test_melSpect8[i] = librosa.util.normalize(test_melSpect8[i])
for i in range(len(test_melSpect9)):
    test_melSpect9[i] = librosa.util.normalize(test_melSpect9[i])
for i in range(len(test_melSpect10)):
    test_melSpect10[i] = librosa.util.normalize(test_melSpect10[i])

"""plt_pca(train_melSpect, train_labels, "Training melSpect")
plt_pca(test_melSpect5, test_labels5, "Test 5 melSpect")
plt_pca(test_melSpect7, test_labels7, "Test 7 melSpect")
plt_pca(test_melSpect8, test_labels8, "Test 8 melSpect")
plt_pca(test_melSpect9, test_labels9, "Test 9 melSpect")
plt_pca(test_melSpect10, test_labels10, "Test 10 melSpect")"""


#Loading Chromagrams

train_chroma = pickle.load(open('Train/chroma_train.pkl', "rb"))
test_chroma5 = pickle.load(open('Test/chroma_test5.pkl', "rb"))
test_chroma7 = pickle.load(open('Test/chroma_test7.pkl', "rb"))
test_chroma8 = pickle.load(open('Test/chroma_test8.pkl', "rb"))
test_chroma9 = pickle.load(open('Test/chroma_test9.pkl', "rb"))
test_chroma10 = pickle.load(open('Test/chroma_test10.pkl', "rb"))


"""plt_pca(train_chroma, train_labels, "Training chroma")
plt_pca(test_chroma5, test_labels5, "Test 5 chroma")
plt_pca(test_chroma7, test_labels7, "Test 7 chroma")
plt_pca(test_chroma8, test_labels8, "Test 8 chroma")
plt_pca(test_chroma9, test_labels9, "Test 9 chroma")
plt_pca(test_chroma10, test_labels10, "Test 10 chroma")"""

#Loading RMS energy stats

train_rmsStat = pickle.load(open('Train/rmsStat_train.pkl', "rb"))
test_rmsStat5 = pickle.load(open('Test/rmsStat_test5.pkl', "rb"))
test_rmsStat7 = pickle.load(open('Test/rmsStat_test7.pkl', "rb"))
test_rmsStat8 = pickle.load(open('Test/rmsStat_test8.pkl', "rb"))
test_rmsStat9 = pickle.load(open('Test/rmsStat_test9.pkl', "rb"))
test_rmsStat10 = pickle.load(open('Test/rmsStat_test10.pkl', "rb"))

"""plt_pca(train_rmsStat, train_labels, "Training rmsStat")
plt_pca(test_rmsStat5, test_labels5, "Test 5 rmsStat")
plt_pca(test_rmsStat7, test_labels7, "Test 7 rmsStat")
plt_pca(test_rmsStat8, test_labels8, "Test 8 rmsStat")
plt_pca(test_rmsStat9, test_labels9, "Test 9 rmsStat")
plt_pca(test_rmsStat10, test_labels10, "Test 10 rmsStat")"""

##################################################################################

#Select the feature to use
"""sel_feat = train_rmsStat
sel_test5 = test_rmsStat5
sel_test7 = test_rmsStat7
sel_test8 = test_rmsStat8
sel_test9 = test_rmsStat9
sel_test10 = test_rmsStat10"""

"""sel_feat = train_melSpect
sel_test5 = test_melSpect5
sel_test7 = test_melSpect7
sel_test8 = test_melSpect8
sel_test9 = test_melSpect9
sel_test10 = test_melSpect10"""

sel_feat = train_mfccs
sel_test5 = test_mfccs5
sel_test7 = test_mfccs7
sel_test8 = test_mfccs8
sel_test9 = test_mfccs9
sel_test10 = test_mfccs10

"""sel_feat = train_mfccs25
sel_test5 = test5_mfccs25
sel_test7 = test7_mfccs25
sel_test8 = test8_mfccs25
sel_test9 = test9_mfccs25
sel_test10 = test10_mfccs25"""


"""sel_feat=train_chroma
sel_test5=test_chroma5
sel_test7=test_chroma7
sel_test8=test_chroma8
sel_test9=test_chroma9
sel_test10=test_chroma10"""

"""#For Random Forest only
sel_feat = sel_feat.reshape((sel_feat.shape[0],sel_feat.shape[1]*sel_feat.shape[2]))
sel_test5 = sel_test5.reshape((sel_test5.shape[0],sel_test5.shape[1]*sel_test5.shape[2]))
sel_test7 = sel_test7.reshape((sel_test7.shape[0],sel_test7.shape[1]*sel_test7.shape[2]))
sel_test8 = sel_test8.reshape((sel_test8.shape[0],sel_test8.shape[1]*sel_test8.shape[2]))
sel_test9 = sel_test9.reshape((sel_test9.shape[0],sel_test9.shape[1]*sel_test9.shape[2]))
sel_test10 = sel_test10.reshape((sel_test10.shape[0],sel_test10.shape[1]*sel_test10.shape[2]))"""


"""sel_feat =np.concatenate((sel_feat,train_melSpect),axis=1)
sel_test5 =np.concatenate((sel_test5,test_melSpect5),axis=1)
sel_test7 =np.concatenate((sel_test7,test_melSpect7),axis=1)
sel_test8 =np.concatenate((sel_test8,test_melSpect8),axis=1)
sel_test9 =np.concatenate((sel_test9,test_melSpect9),axis=1)
sel_test10 =np.concatenate((sel_test10,test_melSpect10),axis=1)"""

"""sel_feat = np.concatenate((sel_feat,train_chroma),axis=1)
sel_test5 = np.concatenate((sel_test5,test_chroma5),axis=1)
sel_test7 = np.concatenate((sel_test7,test_chroma7),axis=1)
sel_test8 = np.concatenate((sel_test8,test_chroma8),axis=1)
sel_test9 = np.concatenate((sel_test9,test_chroma9),axis=1)
sel_test10 = np.concatenate((sel_test10,test_chroma10),axis=1)"""


#PCA Analysis if needed
"""pca = PCA(n_components=PCA_COMPONENT, svd_solver='full')
sel_feat = pca.fit_transform(sel_feat.reshape(-1, sel_feat.shape[-1])).reshape((sel_feat.shape[0],sel_feat.shape[1],PCA_COMPONENT))
sel_test5 = pca.fit_transform(sel_test5.reshape(-1, sel_test5.shape[-1])).reshape((sel_test5.shape[0],sel_test5.shape[1],PCA_COMPONENT))
sel_test7 = pca.fit_transform(sel_test7.reshape(-1, sel_test7.shape[-1])).reshape((sel_test7.shape[0],sel_test7.shape[1],PCA_COMPONENT))
sel_test8 = pca.fit_transform(sel_test8.reshape(-1, sel_test8.shape[-1])).reshape((sel_test8.shape[0],sel_test8.shape[1],PCA_COMPONENT))
sel_test9 = pca.fit_transform(sel_test9.reshape(-1, sel_test9.shape[-1])).reshape((sel_test9.shape[0],sel_test9.shape[1],PCA_COMPONENT))
sel_test10 = pca.fit_transform(sel_test10.reshape(-1, sel_test10.shape[-1])).reshape((sel_test10.shape[0],sel_test10.shape[1],PCA_COMPONENT))


#print(pca.explained_variance_ratio_)"""

params = {'dim': (sel_feat.shape),
          'batch_size': BATCH_SIZE,
          'n_classes': 10,
          'n_channels': N_CHANNELS,
          'shuffle': True,
          'validation_split': 0.1,
          'n_epochs': 50,
          'lr': 0.001}

print(params['dim'],params['batch_size'],params['lr'])


#First model, RF
"""for j in [50,100,200,500]:
    clf = RandomForestClassifier(n_estimators=j, random_state=0).fit(sel_feat, train_labels)
    test_scores = []
    score = clf.score(sel_test5, test_labels5) #accuracy
    test_scores.append(score)
    y_pred5 = clf.predict(sel_test5)
    score = clf.score(sel_test7, test_labels7) #accuracy
    test_scores.append(score)
    y_pred7 = clf.predict(sel_test7)
    score = clf.score(sel_test8, test_labels8) #accuracy
    test_scores.append(score)
    y_pred8 = clf.predict(sel_test8)
    score = clf.score(sel_test9, test_labels9) #accuracy
    test_scores.append(score)
    y_pred9 = clf.predict(sel_test9)
    score = clf.score(sel_test10, test_labels10) #accuracy
    test_scores.append(score)
    y_pred10 = clf.predict(sel_test10)

    print(j, test_scores)
    y_pred = np.concatenate((y_pred5,y_pred7,y_pred8,y_pred9,y_pred10),axis=0)
    y_true = np.concatenate((test_labels5,test_labels7,test_labels8,test_labels9,test_labels10),axis=0)
    cm = confusion_matrix(y_true, y_pred)
    plt_confusion_matrix(cm)
"""


# First Neural Network, no Convolution layers 
"""denseModel = tf.keras.Sequential([
    layers.Flatten(input_shape=(params['dim'][1], params['dim'][2], params['n_channels'])),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2048, activation='relu'),
    layers.Dropout(0.5),
    #layers.Dense(params['dim']/2, activation='relu'),
    #layers.Dense(params['n_classes'], activation='Softmax'),
    layers.Dense(params['n_classes'],activation='softmax'),
])

selected_model(denseModel, sel_feat, train_labels,params['lr'] ,params['batch_size'], params['n_epochs'], sel_test5, sel_test7, sel_test8, sel_test9, sel_test10, test_labels5, test_labels7, test_labels8, test_labels9, test_labels10)
"""

#Second Neural Network, Convolution layers
"""model2D = tf.keras.Sequential([
        layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(params['dim'][1], params['dim'][2], params['n_channels']),padding='same', strides=1),
        layers.MaxPooling2D(pool_size=(2),padding='same'),
        layers.Dropout(0.5),
        layers.Conv2D(64, kernel_size=3, activation='relu',strides=1,padding='same'),
        layers.MaxPooling2D(pool_size=(2),padding='same'),
        layers.Dropout(0.3),
        layers.Conv2D(64, kernel_size=3, activation='relu',strides=1,padding='same'),
        layers.MaxPooling2D(pool_size=(2),padding='same'),
        #layers.Conv2D(128, kernel_size=3, activation='relu',strides=1,padding='same'),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(params['n_classes'], activation='softmax'),
])


selected_model(model2D, sel_feat, train_labels,params['lr'] ,params['batch_size'], params['n_epochs'], sel_test5, sel_test7, sel_test8, sel_test9, sel_test10, test_labels5, test_labels7, test_labels8, test_labels9, test_labels10)"""


#Third Neural Network, transfer learning

#To convert from 1 to 3 channels
sel_feat=np.repeat(sel_feat[..., np.newaxis], 3, axis=3)
sel_test5=np.repeat(sel_test5[..., np.newaxis], 3, axis=3)
sel_test7=np.repeat(sel_test7[..., np.newaxis], 3, axis=3)
sel_test8=np.repeat(sel_test8[..., np.newaxis], 3, axis=3)
sel_test9=np.repeat(sel_test9[..., np.newaxis], 3, axis=3)
sel_test10=np.repeat(sel_test10[..., np.newaxis], 3, axis=3)

vgg16_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top = False, input_shape=(params['dim'][1], params['dim'][2], params['n_channels']))
vgg16_custom_model = tf.keras.Sequential()
   

# Freeze the layers 
for layer in vgg16_custom_model.layers:
    layer.trainable = False
    
vgg16_custom_model.add(layers.Flatten(name='lastFlatten'))
vgg16_custom_model.add(layers.Dense(256, activation='relu'))
vgg16_custom_model.add(layers.Dropout(0.5))
vgg16_custom_model.add(layers.Dense(params['n_classes'], activation='softmax'))



selected_model(vgg16_custom_model, sel_feat, train_labels, params['lr'], params['batch_size'], params['n_epochs'], sel_test5, sel_test7, sel_test8, sel_test9, sel_test10, test_labels5, test_labels7, test_labels8, test_labels9, test_labels10)


