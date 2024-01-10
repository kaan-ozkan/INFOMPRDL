import os
import h5py
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Reshape

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

#preprocessing
#preprocessing
def load_data(folder_paths, batch_size =8):
    data = []
    labels = []
    for folder_path in folder_paths:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.h5'):
                file_path = os.path.join(folder_path, file_name)
                matrix = read_data(file_path)
                parts = file_name.split()
                label = parts[1].split('.')[0] if len(parts) > 1 else None
                data.append(matrix)
                labels.append(label)
                
                if len(data) == batch_size:
                    data = np.array(data)
                    labels = np.array(labels)
                    label_encoder = LabelEncoder()
                    labels = label_encoder.fit_transform(labels)
                    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
                    #scaling ? 
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
                    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
                    yield X_train, X_test, y_train, y_test
                    data = []
                    labels = []

    if data: 
        data = np.array(data)
        labels = np.array(labels)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        #scaling ? 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        yield X_train, X_test, y_train, y_test

def create_cnn_rnn_model(input_shape, num_task_types):
    model = Sequential()

    #add CNN layers here? 
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    #LSTM
    model.add(Reshape((model.output_shape[1], 1)))
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape)) # add activation relu 
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    num_task_types = 4 
    model.add(Dense(num_task_types, activation='softmax'))
    return model

intra_train_path = "Intra/train/"
intra_test_path = "Intra/test/"
X_intra_train, X_intra_test, y_intra_train, y_intra_test = next(load_data([intra_train_path]))

cross_train_path = "Cross/train"
cross_test_paths = ["Cross/test1", "Cross/test2", "Cross/test3"]
(X_cross_train, X_cross_test, y_cross_train, y_cross_test) = next(load_data(["Cross/train"]))
X_cross_tests = [next(load_data([test_path])) for test_path in cross_test_paths]

#had to do that cause apparently the data before wasnt the right format for the LSTM 
X_cross_train_reshaped = X_cross_train.reshape(X_cross_train.shape[0], X_cross_train.shape[2], X_cross_train.shape[1])
X_cross_test_reshaped = X_cross_test.reshape(X_cross_test.shape[0], X_cross_test.shape[2], X_cross_test.shape[1])
X_intra_train_reshaped = X_intra_train.reshape(X_intra_train.shape[0], X_intra_train.shape[2], X_intra_train.shape[1])
X_intra_test_reshaped = X_intra_test.reshape(X_intra_test.shape[0], X_intra_test.shape[2], X_intra_test.shape[1])

input_shape_cnn = (X_intra_train_reshaped.shape[1], X_intra_train_reshaped.shape[2], 1)
cnn_rnn_model = create_cnn_rnn_model(input_shape_cnn, num_task_types=4)
cnn_rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = cnn_rnn_model.fit(X_intra_train_reshaped, y_intra_train, epochs=10, batch_size=32, validation_data=(X_intra_test_reshaped, y_intra_test))