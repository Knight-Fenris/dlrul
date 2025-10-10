# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import gc
gc.collect()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import h5py
import time
# import matplotlib
# import seaborn as sns
# from pandas import DataFrame
import matplotlib.pyplot as plt
# from matplotlib import gridspec
from pprint import pprint
# %matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
all_files=[]
for dirname, _, filenames in os.walk('./data_set'):
    for filename in filenames:
        all_files.append(os.path.join(dirname, filename))
# print(all_files)
all_files.sort()
pprint(all_files)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
from tensorflow import keras
# import keras
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout, Activation, Masking

import tensorflow as tf
from tensorflow.keras.metrics import R2Score as RSquare
from sklearn.preprocessing import StandardScaler

# %%
# Initialize the MirroredStrategy to use both T4 GPUs
# try:
#     strategy = tf.distribute.MirroredStrategy()
#     print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# except ValueError:
#     # If MirroredStrategy is not available, use the default strategy.
#     strategy = tf.distribute.get_strategy()
#     print("Single device strategy chosen.")

# import tensorflow as tf

# tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.test.gpu_device_name()

# %%
#other imports
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearnex import patch_sklearn
# patch_sklearn()

# %%
def read_hdf(filename):
        # filename='D:\\Katinder\\af\\nasa\\new_dataset\\data_set\\N-CMAPSS_DS01-005.h5'

        # Time tracking, Operation time (min):  0.003
        t = time.process_time()  

        # Load data
        with h5py.File(filename, 'r') as hdf:
                # Development set
                W_dev = np.array(hdf.get('W_dev'))             # W
                X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s= sensors
                X_v_dev = np.array(hdf.get('X_v_dev'))         # X_v= virtual sensors
                T_dev = np.array(hdf.get('T_dev'))             # T
                Y_dev = np.array(hdf.get('Y_dev'))             # Y= RUL  
                A_dev = np.array(hdf.get('A_dev'))             # Auxiliary

                # Test set
                W_test = np.array(hdf.get('W_test'))           # W
                X_s_test = np.array(hdf.get('X_s_test'))       # X_s
                X_v_test = np.array(hdf.get('X_v_test'))       # X_v
                T_test = np.array(hdf.get('T_test'))           # T
                Y_test = np.array(hdf.get('Y_test'))           # RUL  
                A_test = np.array(hdf.get('A_test'))           # Auxiliary
                
                # Varnams
                W_var = np.array(hdf.get('W_var'))
                X_s_var = np.array(hdf.get('X_s_var'))  
                X_v_var = np.array(hdf.get('X_v_var')) 
                T_var = np.array(hdf.get('T_var'))
                A_var = np.array(hdf.get('A_var'))
                
                # from np.array to list dtype U4/U5
                W_var = list(np.array(W_var, dtype='U20'))
                X_s_var = list(np.array(X_s_var, dtype='U20'))  
                X_v_var = list(np.array(X_v_var, dtype='U20')) 
                T_var = list(np.array(T_var, dtype='U20'))
                A_var = list(np.array(A_var, dtype='U20'))
                                
        ## Train data
        df_a_dev=pd.DataFrame(A_dev,columns=A_var)
        df_t_dev=pd.DataFrame(T_dev,columns=T_var)
        df_w_dev=pd.DataFrame(W_dev,columns=W_var)
        df_xs_dev=pd.DataFrame(X_s_dev,columns=X_s_var)
        df_xv_dev=pd.DataFrame(X_v_dev,columns=X_v_var)
        df_y_dev=pd.DataFrame(Y_dev,columns=['RUL'])

        #get rul
        trainy=df_y_dev[['RUL']]

        #get x from sensors values
        trainx=pd.concat([df_xs_dev,df_w_dev],axis=1)

        print(type(trainy),trainy.shape, type(trainx), trainx.shape)

        ## Test data
        df_a_test=pd.DataFrame(A_test,columns=A_var)
        df_t_test=pd.DataFrame(T_test,columns=T_var)
        df_w_test=pd.DataFrame(W_test,columns=W_var)
        df_xs_test=pd.DataFrame(X_s_test,columns=X_s_var)
        df_xv_test=pd.DataFrame(X_v_test,columns=X_v_var)
        df_y_test=pd.DataFrame(Y_test,columns=['RUL'])

        #get health state from aux
        testy=df_y_test[['RUL']]

        #get x from sensors values
        testx=pd.concat([df_xs_test,df_w_test],axis=1)

        print(type(testy),testy.shape, type(testx), testx.shape)
                

        # W = np.concatenate((W_dev, W_test), axis=0)  
        # X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
        # X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
        # T = np.concatenate((T_dev, T_test), axis=0)
        # Y = np.concatenate((Y_dev, Y_test), axis=0) 
        # A = np.concatenate((A_dev, A_test), axis=0) 
        
        print('')
        print("Operation time (min): " , (time.process_time()-t)/60)
        print('')
        # print ("W shape: " + str(W.shape))
        # print ("X_s shape: " + str(X_s.shape))
        # print ("X_v shape: " + str(X_v.shape))
        # print ("T shape: " + str(T.shape))
        # print ("A shape: " + str(A.shape))

        return (trainx,trainy,df_a_dev,testx,testy,df_a_test)

# %%
curr_file=all_files[2]
print(curr_file)

# %%
trainx,trainy,adev,testx,testy,atest = read_hdf(filename=curr_file)

# %% [markdown]
# ### Using Windowing per unit

# %%
trainx.info(show_counts=True)

# %%
#standard scale
# standard scale

sc=StandardScaler()
sc.fit(trainx.values)
trainx_l_sc=sc.transform(trainx.values)
testx_l_sc=sc.transform(testx.values)

# ysc=StandardScaler()
# ysc.fit(trainy.values.reshape(-1, 1))
# trainy_l_sc=ysc.transform(trainy.values.reshape(-1, 1))
# testy_l_sc=ysc.transform(testy.values.reshape(-1, 1))

print(trainx_l_sc.shape, testx_l_sc.shape, trainy.shape, testy.shape)

# turn back to dataframe with col headers
trainx_l_sc=pd.DataFrame(trainx_l_sc,columns=trainx.columns)
# trainy_l_sc=pd.DataFrame(trainy_l_sc,columns=['RUL'])
testx_l_sc=pd.DataFrame(testx_l_sc,columns=trainx.columns)
# testy_l_sc=pd.DataFrame(testy_l_sc,columns=['RUL'])

print(trainx_l_sc.shape, testx_l_sc.shape, trainy.shape, testy.shape)

type(trainx_l_sc),type(testy)

# %%
# # max = trainy.max()ti
# trainy *= max
# trainy = trainy.astype(np.int32)

# %%
#from the entire combined set take units one by one and generate the windowed data: samples and labels

# function for windowed samples: adapted from https://github.com/mohyunho/N-CMAPSS_DL/blob/main/utils/data_preparation_unit.py 

def time_window_slicing_sample (input_array, window_length, unit, sequence_cols, stride=1):

    window_lst = []  # a python list to hold the windows
    input_temp = input_array[input_array['unit'] == unit][sequence_cols].values
    print ("Unit%s input array shape: " %unit, input_temp.shape)
    num_samples = int((input_temp.shape[0] - window_length)/stride) + 1

    for i in range(num_samples):
        window = input_temp[i*stride:i*stride + window_length,:]  # each individual window
        window_lst.append(window)

    sample_array = np.dstack(window_lst).astype(np.float32)
    print ("sample_array.shape", sample_array.shape)
    return sample_array

# function for windowed labels: same source

def time_window_slicing_label (input_array, window_length, unit, sequence_cols = 'RUL', stride=1):

    window_lst = []  # a python list to hold the windows
    input_temp = input_array[input_array['unit'] == unit][sequence_cols].values
    num_samples = int((input_temp.shape[0] - window_length)/stride) + 1
    for i in range(num_samples):
        window = input_temp[i*stride:i*stride + window_length]  # each individual window
        window_lst.append(window) # length of window variable 

    label_array = np.asarray(window_lst).astype(np.float32)
    return label_array[:,-1]

# %%
# for batch in train_gen:
#     print(batch[1].shape, batch[0].shape)
#     break

# %%
# df.values[0*1:0*1 + 50,:].shape

# %%
trainy['RUL'].unique()

# %%
trainy = trainy[trainy!=0].dropna()
trainx_l_sc = trainx_l_sc.loc[trainy.index]
testy = testy[testy!=0].dropna()
testx_l_sc = testx_l_sc.loc[testy.index]

print(trainx_l_sc.shape, trainy.shape, testy.shape, testx_l_sc.shape)

# %%
df=trainx_l_sc.copy()
df['unit']=adev['unit']
df['RUL']=trainy['RUL']
adev.unit.unique()

# %%
# get unique units from the train data
all_units_data=[]
for un in adev.unit.unique():
    all_units_data.append(time_window_slicing_sample(df, 50, un, df.columns.difference(['unit','RUL']), 25))

all_units_data=np.dstack(all_units_data)

# %%
all_units_data=all_units_data.transpose(2,0,1)
all_units_data.shape

# %%
# label slicing
all_labels=[]
for un in adev.unit.unique():
    all_labels=all_labels+time_window_slicing_label(df, 50, un, stride=25).tolist()

print(len(all_labels))
all_labels=np.array(all_labels)

# %%
np.unique(all_labels)

# %%
#test data
df=testx_l_sc.copy()
df['unit']=atest['unit']
df['RUL']=testy['RUL']

# get unique units from the test data
test_units_data=[]
for un in atest.unit.unique():
    test_units_data.append(time_window_slicing_sample(df, 50, un, df.columns.difference(['unit','RUL']), 25))

test_units_data=np.dstack(test_units_data)
test_units_data=test_units_data.transpose(2,0,1)
print(test_units_data.shape)

# label slicing
test_labels=[]
for un in atest.unit.unique():
    test_labels=test_labels+time_window_slicing_label(df, 50, un, stride=25).tolist()

print(len(test_labels))
test_labels=np.array(test_labels)

# %%
# train val split: 90-10%
from sklearn.model_selection import train_test_split
xtrain,xval,ytrain,yval=train_test_split(all_units_data,all_labels,test_size=0.2)

# xtrain = all_units_data[ : int(all_units_data.shape[0]*0.9)+1, : , : ]
# xval = all_units_data[int(all_units_data.shape[0]*0.9)+1 : , : , : ]

# ytrain = all_labels[ : int(all_labels.shape[0]*0.9)+1]
# yval = all_labels[int(all_labels.shape[0]*0.9)+1 : ]

# %%
ytrain = ytrain.reshape([-1, 1])
yval = yval.reshape([-1, 1])
test_labels = test_labels.reshape([-1, 1])

# %%
ytrain.shape, yval.shape

# %%
# batch_size= 2
# train_data = tf.data.Dataset.from_tensor_slices((all_units_data, all_labels))
# train_data = train_data.batch(batch_size).prefetch(1)

# %%
del all_units_data, all_labels, adev, trainx, trainy, trainx_l_sc

# %%
# # taken from SO

# from tensorflow.keras.utils import Sequence

# class DataGenerator(Sequence):
#     def __init__(self, x_set, y_set, batch_size):
#         self.x, self.y = x_set, y_set
#         self.batch_size = batch_size

#     def __len__(self):
#         return int(np.ceil(len(self.x) / float(self.batch_size)))

#     def __getitem__(self, idx):
#         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
#         return batch_x, batch_y

# batch = 8192
# train_gen = DataGenerator(xtrain, ytrain, batch)
# val_gen = DataGenerator(xval, yval, batch)
# test_gen = DataGenerator(test_units_data, test_labels, batch)
# del xtrain, ytrain, xval, yval,
## test_labels, test_units_data

# val_gen.__len__(), val_gen.__getitem__(1)[0].shape, val_gen.__getitem__(1)[1].shape




# %% [markdown]
# ## TF DATASETS
# 

# %%
# BATCH_SIZE will be the global batch size. MirroredStrategy will split it across the GPUs.
# For example, 8192 becomes 4096 per GPU.
# BATCH_SIZE = 8192
BATCH_SIZE = 4096
BUFFER_SIZE = 20000 
# from sklearn.preprocessing import MinMaxScaler
# ytrain = ytrain.reshape(-1, 1)
# yval = yval.reshape(-1, 1)
# test_labels = test_labels.reshape(-1, 1)

# # Create and fit the scaler ONLY on the training data
# # y_scaler = StandardScaler()
# y_scaler = MinMaxScaler()
# ytrain = y_scaler.fit_transform(ytrain)

# # Transform the validation and test data with the SAME scaler
# yval = y_scaler.transform(yval)
# test_labels_scaled = y_scaler.transform(test_labels)

# Create tf.data.Dataset objects. This is the modern and most efficient way.
# tf.data.Dataset.from_tensor_slices creates a dataset from your numpy arrays in memory.
# Before creating your dataset, ensure your numpy arrays are float32
xtrain = xtrain.astype(np.float32)
ytrain = ytrain.astype(np.float32)
xval = xval.astype(np.float32)
yval = yval.astype(np.float32)

# Now create the dataset
# train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
# ...
train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
val_ds = tf.data.Dataset.from_tensor_slices((xval, yval))
test_ds = tf.data.Dataset.from_tensor_slices((test_units_data, test_labels))


# AUTOTUNE lets TensorFlow find the best parallel settings for data loading
AUTOTUNE = tf.data.AUTOTUNE

# --- Setup the Training Pipeline ---
# .shuffle() is important for training to prevent the model from learning the order of the data.
# .batch() groups the data into batches. drop_remainder is often a good idea for distributed training.
# .prefetch(AUTOTUNE) is the key performance optimization. It prepares the next batch(es) 
# on the CPU while the GPUs are busy training on the current batch.
train_gen = (
    train_ds
    .cache()
    .shuffle(buffer_size=BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

# --- Setup the Validation and Test Pipelines ---
# We don't need to shuffle validation or test data.
val_gen = (
    val_ds
    .cache()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

test_gen = (
    test_ds
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

# You can now pass these `tf.data.Dataset` objects directly to tuner.search() and model.fit()
# For example: tuner.search(train_ds, validation_data=val_ds, ...)

# Clean up memory
# del xtrain, ytrain, xval, yval, test_units_data, test_labels
gc.collect()

print("✅ tf.data pipelines created successfully.")

# %% [markdown]
# ### Residual LSTM

# %%
# class ResidualWrapper(tf.keras.Model):
#   def __init__(self, model):
#     super().__init__()
#     self.model = model

#   def call(self, inputs, *args, **kwargs):
#     delta = self.model(inputs, *args, **kwargs)

#     # The prediction for each time step is the input
#     # from the previous time step plus the delta
#     # calculated by the model.
#     return inputs + delta

# MAX_EPOCHS = 50

# def compile_and_fit(model, train, val, patience=4):
#   early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                     patience=patience,
#                                                     mode='min')

#   model.compile(loss=tf.keras.losses.MeanSquaredError(),
#                 optimizer=tf.keras.optimizers.Adam(),
#                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
#   # print(model.predict(np.zeros((1, 50, 18))))
#   # history = model.fit(train, epochs=MAX_EPOCHS,
#   #                     validation_data=val,
#   #                     callbacks=[early_stopping])
#   # return history
#   return model

# %%time
# residual_lstm = ResidualWrapper(
#     tf.keras.Sequential([
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     tf.keras.layers.Dense(
#         1,
#         # The predicted deltas should start small.
#         # Therefore, initialize the output layer with zeros.
#         # kernel_initializer=tf.initializers.zeros()
#         )
# ]))

# model = compile_and_fit(residual_lstm, train_gen, val_gen)
# # history = compile_and_fit(residual_lstm, train_gen, val_gen)

# # IPython.display.clear_output()
# # val_performance['Residual LSTM'] = residual_lstm.evaluate(val_gen, return_dict=True)
# # performance['Residual LSTM'] = residual_lstm.evaluate(test_gen, verbose=0, return_dict=True)
# # print()

# model.build(input_shape=(100, 50, 18))

# y = model.predict(np.zeros((100, 50, 18)))

# y.shape

# %% [markdown]
# ## Transformer

# %%
import tensorflow as tf
from tensorflow.keras import layers


def transformer_encoder(inputs, head_size, num_heads, ff_dim1, dropout, ff_dim2):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim1, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x + res
    x = layers.Conv1D(filters=ff_dim2, kernel_size= 3, activation= 'relu')(x)

    return x

# %%
def transformer_encoder_v2(inputs, head_size, num_heads, ff_dim, dropout):
    """A transformer block with Pre-Layer Normalization."""
    
    # --- Attention Block (with Pre-LN) ---
    # Normalize the inputs *before* they go into the attention layer.
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    # Add the residual connection.
    res = x + inputs

    # --- Feed Forward Block (with Pre-LN) ---
    # Normalize the result of the attention block *before* the feed-forward layers.
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    # Add the second residual connection.
    return x + res

# %%
# !pip -qq install keras_nlp

# %%
# import keras_nlp
# import tensorflow as tf
# import keras
# from keras import layers

# def pos_encoder(inputs, head_size, num_heads, dropout):
#     x = keras_nlp.layers.PositionEmbedding(sequence_length=50)(inputs)
#     # Attention and Normalization
#     x = layers.MultiHeadAttention(
#         key_dim=head_size,
#         num_heads=num_heads,
#         dropout=dropout
#     )(x, x)
#     x = layers.Dropout(dropout)(x)
#     return x

# %%

# def build_model2(head_size = 10, num_heads = 5):
#     input_shape= (50, 18)
#     inputs = tf.keras.Input(shape= input_shape)
#     x = inputs
#     x = pos_encoder(x, head_size, num_heads, 0.0)
#     x = layers.GlobalAveragePooling1D(data_format='channels_last')(x)
#     x = layers.Dense(30, activation='selu')(x)
#     x = layers.Dropout(0.25)(x)
#     outputs = layers.Dense(1, activation='linear')(x)
    
#     model = tf.keras.Model(inputs, outputs)
#     model.compile(loss = 'mean_squared_error', optimizer=tf.keras.optimizers.AdamW(), metrics=['mean_absolute_error'])
#     return model

# %%
# model1 = build_model2()
# model1.summary()

# %%
import keras_nlp

def build_model(hp):
    input_shape = (50, 18)    
    head_size= hp.Int("head_size", min_value= 2, max_value=10, step=1)
    num_heads = hp.Int("num_heads", min_value=1, max_value=5, step=1)
    ff_dim= hp.Int("ff_dim", min_value= 5, max_value=32, step=1)
#     ff_dim2= hp.Int("ff_dim2", min_value= 5, max_value=32, step=1)
    num_transformer_blocks= hp.Int("num_blocks", min_value=1, max_value= 10, step=1)
    mlp_units= hp.Int("mlp_units", min_value=1, max_value=5, step=1)
    # head_size = hp.Int("head_size", min_value=16, max_value=64, step=16)
    # num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)
    # ff_dim = hp.Int("ff_dim", min_value=64, max_value=256, step=64)
    # num_transformer_blocks = hp.Int("num_blocks", min_value=1, max_value=5, step=1)
    # mlp_units = hp.Int("mlp_units", min_value=1, max_value=5, step=1)
    learning_rate = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4,1e-5,5e-5,1e-6]) # Tuning LR is key!

    # --- Model Architecture ---
    inputs = tf.keras.Input(shape=input_shape)
    x = keras_nlp.layers.PositionEmbedding(sequence_length=input_shape[0])(inputs)

    for i in range(num_transformer_blocks):
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.3, step=0.1)
        x = transformer_encoder_v2(x, head_size, num_heads, ff_dim, dropout_rate)
    
    x = layers.Lambda(lambda seq: seq[:, -1, :], name="extract_last_timestep")(x)
    
    for i in range(mlp_units):
        dim = hp.Int(f'dim_{i}', min_value=32, max_value=128, step=32)
        mlp_dropout = hp.Float(f'mlp_dropout_{i}', min_value=0.1, max_value=0.4, step=0.1)
        x = layers.Dense(dim, activation='selu')(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    outputs = layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs)

    # --- Compile Step ---
    if hp.Choice('optimizer', ['adam', 'adamw']) == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mean_absolute_error"])
    return model

# %%
# with strategy.scope():
import keras_nlp
def build_model_old(
    hp   
):
    input_shape= (50, 18)
    head_size= hp.Int("head_size", min_value= 2, max_value=10, step=1)
    num_heads = hp.Int("num_heads", min_value=1, max_value=5, step=1)
    ff_dim1= hp.Int("ff_dim1", min_value= 5, max_value=32, step=1)
    ff_dim2= hp.Int("ff_dim2", min_value= 5, max_value=32, step=1)
    num_transformer_blocks= hp.Int("num_blocks", min_value=1, max_value= 10, step=1)
    mlp_units= hp.Int("mlp_units", min_value=1, max_value=5, step=1)
    inputs = tf.keras.Input(shape=input_shape)
    
    seq_length = input_shape[0]
    embed_dim = input_shape[1]
    #Positional Embeddings
    
    # positions = tf.range(start=0, limit=seq_length, delta=1)
    # position_embedding_layer = layers.Embedding(input_dim=seq_length, output_dim=embed_dim)
    # position_embeddings = position_embedding_layer(positions)
    # x = inputs + position_embeddings

    # x = keras_nlp.layers.PositionEmbedding(sequence_length=input_shape[0])(inputs)
    x=inputs
    #End
    for _ in range(num_transformer_blocks):
        dropout = hp.Choice(f'dropout_{_}', [0.0, 0.1, 0.25, 0.5])
        x = transformer_encoder(x, head_size, num_heads, ff_dim1, dropout, ff_dim2)
        # x = transformer_encoder_v2(x, head_size, num_heads, ff_dim1, dropout, ff_dim2)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    # x = layers.Lambda(lambda seq: seq[:, -1, :])(x)
    for _ in range(mlp_units):
        dim = hp.Int(f'dim_{_}', min_value=10, max_value=50, step=2)
        mlp_dropout = hp.Choice(f'mlp_dropout_{_}', [0.0, 0.1, 0.25, 0.5])
        mlp_activation = hp.Choice(f'mlp_activ_{_}', ['relu', 'sigmoid', 'tanh', 'selu', 'elu', 'silu'])
        x = layers.Dense(dim, activation=mlp_activation)(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(1, activation="linear")(x)  # Single output for regression

    model = tf.keras.Model(inputs, outputs)
    optimizer = hp.Choice('optimizer', ['sgd', 'rmsprop', 'adam', 'adamw', 'adagrad', 'adamax'])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                # optimizer=optimizer,
                optimizer=tf.keras.optimizers.Adam(),

                metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

# %%
def build_model2(
    hp   
):
    input_shape= (50, 18)
    head_size= hp.Int("head_size", min_value= 2, max_value=10, step=1)
    num_heads = hp.Int("num_heads", min_value=1, max_value=5, step=1)
    ff_dim1= hp.Int("ff_dim1", min_value= 5, max_value=32, step=1)
    ff_dim2= hp.Int("ff_dim2", min_value= 5, max_value=32, step=1)
    num_transformer_blocks= 1
    mlp_units= 1
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        dropout = hp.Choice(f'dropout_{_}', [0.0, 0.1, 0.25, 0.5])
        x = transformer_encoder(x, head_size, num_heads, ff_dim1, dropout, ff_dim2)
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for _ in range(mlp_units):
        dim = hp.Int(f'dim_{_}', min_value=10, max_value=50, step=2)
        mlp_dropout = hp.Choice(f'mlp_dropout_{_}', [0.0, 0.1, 0.25, 0.5])
        mlp_activation = hp.Choice(f'mlp_activ_{_}', ['relu', 'sigmoid', 'tanh', 'selu', 'elu', 'silu'])
        x = layers.Dense(dim, activation=mlp_activation)(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(1, activation="linear")(x)  # Single output for regression

    model = tf.keras.Model(inputs, outputs)
    optimizer = hp.Choice('optimizer', ['sgd', 'rmsprop', 'adam', 'adamw', 'adagrad', 'adamax'])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

# %%
import keras_tuner
# with strategy.scope():
build_model(keras_tuner.HyperParameters())

# %%
# !rm -rf ./keras_tuner

# %%
# with strategy.scope():
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_mean_absolute_error",
    max_trials=100, #100
    executions_per_trial=2,
    overwrite=False,
    directory="keras_tuner",
    project_name="transformers_2",
    #Uncomment for t4x2
    # distribution_strategy=strategy 
)

# %%
tuner.search_space_summary()

# %%
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import numpy as np

class RealMAECallback(tf.keras.callbacks.Callback):
    """
    A custom callback to calculate the unscaled MAE and add it
    to the Keras logs for proper printing.
    """
    def __init__(self, x_val, y_val_unscaled, y_scaler):
        super().__init__()
        self.x_val = x_val
        self.y_val_unscaled = y_val_unscaled
        self.y_scaler = y_scaler

    def on_epoch_end(self, epoch, logs=None):
        # Make predictions and inverse transform as before
        predictions_scaled = self.model.predict(self.x_val, verbose=0)
        predictions_original = self.y_scaler.inverse_transform(predictions_scaled)
        
        # Calculate the real MAE
        mae = mean_absolute_error(self.y_val_unscaled, predictions_original)
        
        # ✅ Add the metric to the logs dictionary. Keras will print it automatically.
        if logs is not None:
            logs['val_mae_original'] = mae
# callbacks.append(RealMAECallback)
# final_callbacks+=[RealMAECallback]


# %%
import os
from tensorflow.keras.callbacks import Callback

class NVIDIASmiMonitor(Callback):
    """A Keras callback to print nvidia-smi output at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n--- GPU Status at end of Epoch {epoch + 1} ---")
        os.system('nvidia-smi')
        print("--------------------------------------------------")
# real_mae_callback = RealMAECallback(xval, yval, y_scaler)



callbacks = [keras.callbacks.EarlyStopping(patience= 20, restore_best_weights= True), keras.callbacks.ReduceLROnPlateau(factor=0.5, patience= 6)]
            #  , real_mae_callback]
# callbacks += [real_mae_callback]
gpu_monitor = NVIDIASmiMonitor()
final_callbacks = callbacks + [gpu_monitor]

# %%
print(callbacks
      )

# %%
tf.config.list_physical_devices()

# %%
# for batch in train_gen:
#     print(batch[0].shape)
#     print(batch[1].shape)

#DO NOT UNCOMMENT, HANGS THE NOTEBOOK

# %% [markdown]
# ## Trial for T4x2

# %%
# # --- UPDATED ISOLATION TEST CELL ---

# print("Starting isolation test (v2) to bypass Keras Tuner...")

# # Temporarily create new datasets for this test WITHOUT drop_remainder=True
# # This allows the test to run even if the dataset is smaller than the batch size.
# test_train_ds = train_gen
# test_val_ds = val_gen


# # Re-define the monitor callback if it's not in scope
# if 'NVIDIASmiMonitor' not in globals():
#     class NVIDIASmiMonitor(Callback):
#         def on_epoch_end(self, epoch, logs=None):
#             print(f"\n--- GPU Status at end of Epoch {epoch + 1} ---")
#             os.system('nvidia-mi')
#             print("--------------------------------------------------")

# # 1. Enter the strategy scope
# with strategy.scope():
#     # ... (The model-building code is exactly the same as before)
#     print("Building a single, non-tuned model inside strategy scope...")
#     input_shape = (50, 18)
#     inputs = tf.keras.Input(shape=input_shape)
#     x = keras_nlp.layers.PositionEmbedding(sequence_length=input_shape[0])(inputs)
#     for _ in range(4):
#         x = transformer_encoder(x, head_size=8, num_heads=4, ff_dim1=32, dropout=0.1, ff_dim2=32)
#     x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
#     x = layers.Dense(32, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)
#     outputs = layers.Dense(1, activation="linear")(x)
#     isolated_model = tf.keras.Model(inputs, outputs)
    
#     print("Compiling the isolated model...")
#     isolated_model.compile(
#         loss=tf.keras.losses.MeanSquaredError(),
#         optimizer=tf.keras.optimizers.Adam(),
#         metrics=[tf.keras.metrics.MeanAbsoluteError()]
#     )
#     isolated_model.summary()

# # 4. Fit the model using the NEW test datasets
# print("\nStarting training for the isolated model...")
# history_isolated = isolated_model.fit(
#     test_train_ds,          # Use the new test dataset
#     validation_data=test_val_ds, # Use the new test dataset
#     epochs=5,
#     callbacks=[NVIDIASmiMonitor()]
# )

# print("Isolation test complete.")

# %% [markdown]
# ## Tuner
# 

# %%
#Epochs 100
history = tuner.search(train_gen,
                       verbose=2,
                       epochs=100, validation_data=val_gen, callbacks=callbacks)

# %%
# with strategy.scope():
best_hps = tuner.get_best_hyperparameters()[0]
# print(f"""
# The hyperparameter search is complete. The optimal number of units in the first dense layer is {best_hps.get('units_0')} and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}.
# """)

# Build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)

# %% [markdown]
# 

# %%
print("The hyperparameter search is complete. Here are the optimal values:")
for hp_name in best_hps.values.keys():
    print(f"{hp_name}: {best_hps.get(hp_name)}")

# %%
# model = tuner.get_best_models()[0]

# %%
model.summary()

# %%
import keras.backend as K

def score_metric_fn(y_true, y_pred):
    score = 0
    # y_true = y_true.numpy()
    # y_pred = y_pred.numpy()
    for i in range(len(y_pred)):
        if y_true[i] <= y_pred[i]:
            score = score + np.exp(-(y_true[i] - y_pred[i]) / 10.0) - 1
        else:
            score = score + np.exp((y_true[i] - y_pred[i]) / 13.0) - 1
    return score

# %%
# with strategy.scope():

# model.compile(loss= 'mean_squared_error', metrics = ['mean_absolute_percentage_error', score_metric_fn, 'mean_absolute_error'], run_eagerly= True, optimizer='adam')
model.compile(loss= 'mean_squared_error', metrics = ['mean_absolute_percentage_error', 'mean_absolute_error'], run_eagerly= True, optimizer='adam')

# %%
# with strategy.scope():
model.evaluate(test_gen)

# %%
# with strategy.scope():
history = model.fit(train_gen, validation_data= val_gen, epochs= 100, 
                    verbose=2,
                    callbacks= callbacks)

# %%
model.evaluate(test_gen)

# %%
model.predict(val_gen).max()

# %%
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3, amsgrad= True)
# model2 = build_model2()
# model2.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse', tf.keras.metrics.MeanAbsolutePercentageError(), RSquare(), score_metric_fn])
# model2.summary()

# %%
# tf.config.run_functions_eagerly(True)

# %%
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# history=model2.fit(train_gen, validation_data=val_gen, epochs=2, batch_size=1024, verbose=1, callbacks=[callback])

# %%
# test_labels[:-957].shape, ypred.shape
# # train_gen = DataGenerator(xtrain[:-947], ytrain[:-947], 1024)
# # val_gen = DataGenerator(xval[:-787], yval[:-787], 1024)
# # test_gen = DataGenerator(test_units_data[:-957], test_labels[:-957], 1024)

# %%
# max(test_labels)

# %%
# max['RUL']

# %%
# score_metric_fn(ypred, test_labels)

# %%
# #DS04
# print(curr_file)

# # plot predcitions
# ypred=model.predict(test_gen)
# # ypred *= max['RUL']
# # type(ypred),ypred[0]
# plt.figure(figsize=(15,15))
# plt.scatter(test_labels[:], ypred)
# plt.plot(np.arange(test_labels[:].max()+1),'r')
# plt.xlabel('Ground truth')
# plt.ylabel('Predicted RUL')
# plt.show()

# # pprint(model.evaluate(test_gen, return_dict=True))

# #  Extract the loss values and epoch numbers from the history object
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training loss")
# plt.plot(epochs, val_loss_values, marker='o', linestyle='-', label="Val loss")
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# %%
# !conda install graphviz

# %%
# import tensorflow as tf
# tf.keras.utils.plot_model(model, './transformer.png')

# %%
# test_labels.shape

# %%
# #DS02
# print(curr_file)

# # plot predcitions
# ypred=model2.predict_generator(test_gen)
# # type(ypred),ypred[0]
# plt.figure(figsize=(15,15))
# plt.scatter(test_labels, ypred)
# plt.plot(np.arange(test_labels.max()+1),'r')
# plt.xlabel('Ground truth')
# plt.ylabel('Predicted RUL')
# plt.show()

# pprint(model2.evaluate(test_gen, return_dict=True))

# #  Extract the loss values and epoch numbers from the history object
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training loss")
# plt.plot(epochs, val_loss_values, marker='o', linestyle='-', label="Val loss")
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Epoch 17/50
# # 4626/4626 [==============================] - 590s 127ms/step - loss: 84.4317 - mse: 84.4317 - 
# # mean_absolute_percentage_error: 54239748.0000 - r_square: 0.8342 - score_metric_fn: 1231.5195 - 
# # val_loss: 66.1675 - val_mse: 66.1675 - val_mean_absolute_percentage_error: 113068744.0000 - 
# # val_r_square: 0.6067 - val_score_metric_fn: 1231.8501

# %%
# #DS01
# print(curr_file)

# # plot predcitions
# ypred=model2.predict_generator(test_gen)
# # type(ypred),ypred[0]
# plt.figure(figsize=(15,15))
# plt.scatter(test_labels, ypred)
# plt.plot(np.arange(test_labels.max()+1),'r')
# plt.xlabel('Ground truth')
# plt.ylabel('Predicted RUL')
# plt.show()

# pprint(model2.evaluate(test_gen, return_dict=True))

# #  Extract the loss values and epoch numbers from the history object
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training loss")
# plt.plot(epochs, val_loss_values, marker='o', linestyle='-', label="Val loss")
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Epoch 15/50
# # 4313/4313 [==============================] - 531s 123ms/step - loss: 185.1849 - mse: 185.1849 - 
# # mean_absolute_percentage_error: 73503024.0000 - r_square: 0.7441 - score_metric_fn: 2936.6345 - 
# # val_loss: 65.9510 - val_mse: 65.9510 - 
# # val_mean_absolute_percentage_error: 99419768.0000 - val_r_square: 0.7589 - val_score_metric_fn: 986.7029

# %% [markdown]
# ### xs+xv

# %%
# #DS03
# print(curr_file)

# # plot predcitions
# ypred=model2.predict_generator(test_gen)
# # type(ypred),ypred[0]
# plt.figure(figsize=(15,15))
# plt.scatter(test_labels, ypred)
# plt.plot(np.arange(test_labels.max()+1),'r')
# plt.xlabel('Ground truth')
# plt.ylabel('Predicted RUL')
# plt.show()

# pprint(model2.evaluate(test_gen, return_dict=True))

# #  Extract the loss values and epoch numbers from the history object
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training loss")
# plt.plot(epochs, val_loss_values, marker='o', linestyle='-', label="Val loss")
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Epoch 17/50
# # 4897/4897 [==============================] - 623s 127ms/step - loss: 95.1664 - mse: 95.1664 - 
# # mean_absolute_percentage_error: 36870832.0000 - r_square: 0.7798 - score_metric_fn: 1357.0190 - 
# # val_loss: 98.1559 - val_mse: 98.1559 - val_mean_absolute_percentage_error: 384910400.0000 - 
# # val_r_square: 0.8598 - val_score_metric_fn: 1268.4259

# %%
# #DS04
# print(curr_file)

# # plot predcitions
# ypred=model2.predict_generator(test_gen)
# # type(ypred),ypred[0]
# plt.figure(figsize=(15,15))
# plt.scatter(test_labels, ypred)
# plt.plot(np.arange(test_labels.max()+1),'r')
# plt.xlabel('Ground truth')
# plt.ylabel('Predicted RUL')
# plt.show()

# print(model2.evaluate(test_gen, return_dict=True))

# #  Extract the loss values and epoch numbers from the history object
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training loss")
# plt.plot(epochs, val_loss_values, marker='o', linestyle='-', label="Val loss")
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Epoch 7/50
# # 5605/5605 [==============================] - 707s 126ms/step - loss: 243.7038 - mse: 243.7038 - 
# # mean_absolute_percentage_error: 83435312.0000 - r_square: 0.6378 - score_metric_fn: 4075.2810 - 
# # val_loss: 119.2151 - val_mse: 119.2151 - 
# # val_mean_absolute_percentage_error: 134992080.0000 - val_r_square: 0.2935 - val_score_metric_fn: 1932.1097

# %%
# #DS02
# print(curr_file)

# # plot predcitions
# ypred=model2.predict_generator(test_gen)
# # type(ypred),ypred[0]
# plt.figure(figsize=(15,15))
# plt.scatter(test_labels, ypred)
# plt.plot(np.arange(test_labels.max()+1),'r')
# plt.xlabel('Ground truth')
# plt.ylabel('Predicted RUL')
# plt.show()

# pprint(model2.evaluate(test_gen, return_dict=True))

# #  Extract the loss values and epoch numbers from the history object
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training loss")
# plt.plot(epochs, val_loss_values, marker='o', linestyle='-', label="Val loss")
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Epoch 14/50
# # 4626/4626 [==============================] - 566s 122ms/step - loss: 84.0611 - mse: 84.0611 - 
# # mean_absolute_percentage_error: 65333336.0000 - r_square: 0.8349 - score_metric_fn: 1224.9670 - 
# # val_loss: 86.9182 - val_mse: 86.9182 - val_mean_absolute_percentage_error: 105552408.0000 - 
# # val_r_square: 0.4834 - val_score_metric_fn: 1598.3446

# %%
# #DS01
# print(curr_file)

# # plot predcitions
# ypred=model2.predict_generator(test_gen)
# # type(ypred),ypred[0]
# plt.figure(figsize=(15,15))
# plt.scatter(test_labels, ypred)
# plt.plot(np.arange(test_labels.max()+1),'r')
# plt.xlabel('Ground truth')
# plt.ylabel('Predicted RUL')
# plt.show()

# pprint(model2.evaluate(test_gen, return_dict=True))

# #  Extract the loss values and epoch numbers from the history object
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training loss")
# plt.plot(epochs, val_loss_values, marker='o', linestyle='-', label="Val loss")
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Epoch 26/50
# # 4313/4313 [==============================] - 554s 129ms/step - loss: 118.6990 - mse: 118.6990 -
# # mean_absolute_percentage_error: 53192540.0000 - r_square: 0.8360 - score_metric_fn: 1746.1334 -
# # val_loss: 35.7389 - val_mse: 35.7389 - val_mean_absolute_percentage_error: 50772952.0000 - 
# # val_r_square: 0.8693 - val_score_metric_fn: 658.9175

# %%
# #DS07
# print(curr_file)

# # plot predcitions
# ypred=model2.predict_generator(test_gen)
# # type(ypred),ypred[0]
# plt.figure(figsize=(15,15))
# plt.scatter(test_labels, ypred)
# plt.plot(np.arange(test_labels.max()+1),'r')
# plt.xlabel('Ground truth')
# plt.ylabel('Predicted RUL')
# plt.show()

# pprint(model2.evaluate(test_gen, return_dict=True))

# #  Extract the loss values and epoch numbers from the history object
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training loss")
# plt.plot(epochs, val_loss_values, marker='o', linestyle='-', label="Val loss")
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# %%
# #DS06
# print(curr_file)
# pprint(model2.evaluate(test_gen, return_dict=True))

# # Epoch 33/50
# # 3742/3742 [==============================] - 480s 128ms/step - loss: 95.3214 - mse: 95.3214 - 
# # mean_absolute_percentage_error: 31056236.0000 - r_square: 0.8128 - score_metric_fn: 1417.2399 - 
# # val_loss: 8.5878 - val_mse: 8.5878 - val_mean_absolute_percentage_error: 108833408.0000 - 
# # val_r_square: 0.8721 - val_score_metric_fn: 252.1017

# #  Extract the loss values and epoch numbers from the history object
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training loss")
# plt.plot(epochs, val_loss_values, marker='o', linestyle='-', label="Val loss")
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# %%
# #DS05
# print(curr_file)
# model2.evaluate(test_gen, return_dict=True)
# # Epoch 20/50
# # 3824/3824 [==============================] - 494s 129ms/step - loss: 118.5198 - mse: 118.5198 - 
# # mean_absolute_percentage_error: 39639448.0000 - r_square: 0.7859 - score_metric_fn: 1769.0951 - 
# # val_loss: 7.2974 - val_mse: 7.2974 - val_mean_absolute_percentage_error: 162413088.0000 - 
# # val_r_square: 0.8961 - val_score_metric_fn: 220.4187

# %%
# print(curr_file)

# # Extract the loss values and epoch numbers from the history object
# loss_values = history.history['loss']
# val_loss_values = history.history['val_loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-', label="Training loss")
# plt.plot(epochs, val_loss_values, marker='o', linestyle='-', label="Val loss")
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# %% [markdown]
# #### Performance with random splitting and scaled targets

# %%
# print("For DS04 only:")
# model2.evaluate(test_gen, return_dict=True)
# # Epoch 30/50
# # 5605/5605 [==============================] - 725s 129ms/step - loss: 0.0922 - mse: 0.0922 - mae: 0.2208 - r_square: 0.9078 - score_metric_fn: 20.3761 - val_loss: 0.0916 - val_mse: 0.0916 - val_mae: 0.2251 - val_r_square: 0.9086 - val_score_metric_fn: 21.5785

# %%
# print("For DS03 only:")
# model2.evaluate(test_gen, return_dict=True)
# # print("4897/4897 [==============================] - 629s 128ms/step - loss: 0.1003 - mse: 0.1003 - mae: 0.2242 - r_square: 0.8997 - score_metric_fn: 20.7127 - val_loss: 0.1172 - val_mse: 0.1172 - val_mae: 0.2567 - val_r_square: 0.8827 - val_score_metric_fn: 24.6942")

# %%
# print("For DS01 only:")
# model2.evaluate(test_gen, return_dict=True)
# # print("313/4313 [==============================] - 554s 128ms/step - loss: 0.0727 - mse: 0.0727 - mae: 0.1942 - r_square: 0.9273 - score_metric_fn: 17.8862 - val_loss: 0.0778 - val_mse: 0.0778 - val_mae: 0.2077 - val_r_square: 0.9222 - val_score_metric_fn: 19.9155")

# %%
# model2.evaluate(test_gen)

# %%
# import matplotlib.pyplot as plt
# import tensorflow as tf

# # Assuming you have already trained your model and stored the history
# # in a variable named 'history2'

# # Extract the loss values and epoch numbers from the history object
# loss_values = history5.history['loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-')
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()

# %%
# #DS-02, 10% validation
# history4=model2.fit(train_gen, validation_data=val_gen, epochs=6, batch_size=1024, verbose=1)

# %%
# import matplotlib.pyplot as plt
# import tensorflow as tf

# # Assuming you have already trained your model and stored the history
# # in a variable named 'history2'

# # Extract the loss values and epoch numbers from the history object
# loss_values = history4.history['loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-')
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()

# %%
# #evaluate on test
# model2.evaluate(test_gen)

# %%
# import matplotlib.pyplot as plt
# import tensorflow as tf

# # Assuming you have already trained your model and stored the history
# # in a variable named 'history2'

# # Extract the loss values and epoch numbers from the history object
# loss_values = history3.history['loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-')
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()

# %%
# import matplotlib.pyplot as plt
# import tensorflow as tf

# # Assuming you have already trained your model and stored the history
# # in a variable named 'history2'

# # Extract the loss values and epoch numbers from the history object
# loss_values = history2.history['loss']
# epochs = range(1, len(loss_values) + 1)

# # Create the loss vs. epochs plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, marker='o', linestyle='-')
# plt.title('Loss vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()

# %% [markdown]
# 

# %%
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

# %%
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# %%
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

# %%
# tf.keras.backend.clear_session()

# %%
# import tensorflow as tf

# # Set GPU memory growth option before initializing TensorFlow
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#    except RuntimeError as e:
#        print(e)

# # Initialize TensorFlow
# tf.keras.backend.clear_session()

# # Rest of your TensorFlow code

# %%
# tf.keras.backend.clear_session()

# %%
# %cd /kaggle/working
# Assuming 'model' is your trained Keras model
# model.save('best_3.keras')
# 

# %%
# from IPython.display import FileLink
# FileLink(r'best_3.keras')


