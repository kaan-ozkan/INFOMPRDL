import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy import signal


def get_dataset_name(filename_with_dir):
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('.')[:-1]
    dataset_name = ''.join(temp)
    return dataset_name


def read_data(file_path):
    with h5py.File(file_path, 'r') as f:
        dataset_name = list(f.keys())[0]
        matrix = f.get(dataset_name)[:]
    return matrix


# preprocessing
def load_data(folder_path):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.h5'):
            file_path = os.path.join(folder_path, file_name)
            matrix = read_data(file_path)
            matrix = signal.decimate(matrix, 4, axis=1)  # downsampling
            parts = file_name.split('.')
            label = parts[0].split('_')[-3]
            data.append(matrix)
            labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    return data, labels


def create_cnn_rnn_model(input_shape, num_task_types):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1,248,8906), padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
   
    #model.add(Flatten())
    model.summary()
    
    # LSTM wants [batch, timesteps, feature(vec)]
    model.add(Reshape(( 16, 124)))
    model.summary()
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(16))
    model.add(Dropout(0.5))
    model.add(Dense(num_task_types, activation='softmax'))
    model.summary()
    print("Model created")
    return model


intra_train_path = "Intra/train/"
intra_test_path = "Intra/test/"
X_intra_train, y_intra_train = load_data(intra_train_path)
X_intra_test, y_intra_test = load_data(intra_test_path)

cross_train_path = "Cross/train"
cross_test_path = "Cross/test1"   # "Cross/test2" or "Cross/test3"
X_cross_train, y_cross_train = load_data(cross_train_path)
X_cross_test, y_cross_test = load_data(cross_test_path)

# had to do that cause apparently the data before wasnt the right format for the LSTM
X_cross_train = np.transpose(X_cross_train, (0, 2, 1)).reshape((X_cross_train.shape[0],1, X_cross_train.shape[1],X_cross_train.shape[2] ))
X_cross_test = np.transpose(X_cross_test, (0, 2, 1)).reshape((X_cross_test.shape[0],1,X_cross_test.shape[1],X_cross_test.shape[2]))
X_intra_train = np.transpose(X_intra_train, (0, 2, 1)).reshape((X_intra_train.shape[0],1,X_intra_train.shape[1],X_intra_train.shape[2]))
X_intra_test = np.transpose(X_intra_test, (0, 2, 1)).reshape((X_intra_test.shape[0],1,X_intra_test.shape[1],X_intra_test.shape[2]))

input_shape_cnn = (X_intra_train.shape[0], X_intra_train.shape[1],X_intra_train.shape[2])#,1)
cnn_rnn_model = create_cnn_rnn_model(input_shape_cnn, num_task_types=4)
cnn_rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

intra_checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
intra_early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

cross_checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
cross_early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

history_intra = cnn_rnn_model.fit(X_intra_train, y_intra_train, epochs=10,validation_data=(X_intra_test, y_intra_test), callbacks=[intra_checkpoint_cb, intra_early_stopping_cb], verbose=1)
print("Done INTRA")

final_train_accuracy = history_intra.history['accuracy'][-1]
final_val_accuracy = history_intra.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_accuracy * 100:.2f}%")

# Plotting training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_intra.history['loss'], label='Train Loss')
plt.plot(history_intra.history['val_loss'], label='Validation Loss')
plt.title('Intra: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history_intra.history['accuracy'], label='Train Accuracy')
plt.plot(history_intra.history['val_accuracy'], label='Validation Accuracy')
plt.title('Intra: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

print("starting CROSS")
cnn_rnn_model_CROSS = create_cnn_rnn_model((X_cross_train.shape[0], X_cross_train.shape[1],X_cross_train.shape[2]), num_task_types=4)
cnn_rnn_model_CROSS.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_cross = cnn_rnn_model_CROSS.fit(X_cross_train, y_cross_train, epochs=10, validation_data=(X_cross_test, y_cross_test), callbacks=[cross_checkpoint_cb, cross_early_stopping_cb], verbose=1)
print("Done CROSS")

final_train_accuracy = history_cross.history['accuracy'][-1]
final_val_accuracy = history_cross.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_accuracy * 100:.2f}%")

# Plotting training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_cross.history['loss'], label='Train Loss')
plt.plot(history_cross.history['val_loss'], label='Validation Loss')
plt.title('Cross: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history_cross.history['accuracy'], label='Train Accuracy')
plt.plot(history_cross.history['val_accuracy'], label='Validation Accuracy')
plt.title('Cross: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

'''
def create_cnn_model(input_shape, num_task_types):
    model = Sequential()
    #CNN
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(num_task_types, activation='softmax'))
    return model

def create_lstm_model(input_shape, num_task_types):
    model = Sequential()
    #LSTM
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.7))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(num_task_types, activation='softmax'))
    return model

intra_train_path = "Intra/train/"
intra_test_path = "Intra/test/"
X_intra_train, y_intra_train = load_data(intra_train_path)
X_intra_test, y_intra_test = load_data(intra_test_path)
X_intra_train = np.transpose(X_intra_train, (0, 2, 1))
X_intra_test = np.transpose(X_intra_test, (0, 2, 1))

input_shape = (X_intra_train.shape[1], X_intra_train.shape[2], 1)

intra_checkpoint_cnn = ModelCheckpoint("best_model.h5", save_best_only=True)
intra_early_stopping_cnn = EarlyStopping(patience=10, restore_best_weights=True)
intra_checkpoint_lstm = ModelCheckpoint("best_model.h5", save_best_only=True)
intra_early_stopping_lstm = EarlyStopping(patience=10, restore_best_weights=True)

cnn_model = create_cnn_model(input_shape, num_task_types=4)
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_intra_cnn = cnn_model.fit(X_intra_train, y_intra_train, epochs=10, batch_size=16, validation_data=(X_intra_test, y_intra_test), callbacks=[intra_checkpoint_cnn, intra_early_stopping_cnn], verbose=1)

lstm_model = create_lstm_model(input_shape, num_task_types=4)
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_intra_lstm = lstm_model.fit(X_intra_train, y_intra_train, epochs=10, batch_size=16, validation_data=(X_intra_test, y_intra_test), callbacks=[intra_checkpoint_lstm, intra_early_stopping_lstm], verbose=1)


cross_train_path = "Cross/train"
cross_test_path = "Cross/test1"   #"Cross/test2" or "Cross/test3"
X_cross_train, y_cross_train = load_data(cross_train_path)
X_cross_test, y_cross_test = load_data(cross_test_path)
X_cross_train = np.transpose(X_cross_train, (0, 2, 1))
X_cross_test = np.transpose(X_cross_test, (0, 2, 1))

input_shape = (X_cross_train.shape[1], X_cross_train.shape[2], 1)

cross_checkpoint_cnn = ModelCheckpoint("best_model.h5", save_best_only=True)
cross_early_stopping_cnn = EarlyStopping(patience=10, restore_best_weights=True)
cross_checkpoint_lstm = ModelCheckpoint("best_model.h5", save_best_only=True)
cross_early_stopping_lstm = EarlyStopping(patience=10, restore_best_weights=True)

cnn_model = create_cnn_model(input_shape, num_task_types=4)
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cross_cnn = cnn_model.fit(X_cross_train, y_cross_train, epochs=10, batch_size=16, validation_data=(X_cross_test, y_cross_test), callbacks=[cross_checkpoint_cnn, cross_early_stopping_cnn], verbose=1)

lstm_model = create_lstm_model(input_shape, num_task_types=4)
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cross_lstm = lstm_model.fit(X_cross_train, y_cross_train, epochs=10, batch_size=16, validation_data=(X_cross_test, y_cross_test), callbacks=[cross_checkpoint_lstm, cross_early_stopping_lstm], verbose=1)


# Plotting training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_intra_cnn.history['loss'], label='Train Loss')
plt.plot(history_intra_cnn.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history_intra_cnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_intra_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("intra_cnn.png") 

# Plotting training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_intra_lstm.history['loss'], label='Train Loss')
plt.plot(history_intra_lstm.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history_intra_lstm.history['accuracy'], label='Train Accuracy')
plt.plot(history_intra_lstm.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("intra_lstm.png") 

# Plotting training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_cross_cnn.history['loss'], label='Train Loss')
plt.plot(history_cross_cnn.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history_cross_cnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_cross_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("cross_cnn.png") 

# Plotting training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_cross_lstm.history['loss'], label='Train Loss')
plt.plot(history_cross_lstm.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history_cross_lstm.history['accuracy'], label='Train Accuracy')
plt.plot(history_cross_lstm.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("cross_lstm.png") 
'''
