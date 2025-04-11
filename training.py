'''
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.datasets import mnist

#separa os dados de treino e teste
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#DEVEMOS NORMALIZAR para que o formato dos dados seja adequado para CNNs ou GANs
print("*************************** BEFORE NORMALIZATON ***************************\n")
print(x_train[0])
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
print("\n\n\n---------------------------------------------------------------------------\n\n\n")
print("************************** AFTER NORMALIZATON ***************************\n")
print(x_train[0])


#é adicionada uma dimensão ao dataset para que ele seja adequado para CNNs ou GANs (rever)
print("Before Adjusting Dimensions x_train shape: ", x_train.shape)
x_train = np.expand_dims(x_train, axis=-1)
print("After Adjusting Dimensions x_train shape: ", x_train.shape)
'''
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Carrega os dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliza os dados (escala para [-1, 1])
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_test = (x_test.astype(np.float32) - 127.5) / 127.5

# Ajusta as dimensões dos dados para incluir o canal (1 para imagens em escala de cinza)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Cria a arquitetura da CNN
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=x_train.shape[1:]),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    #entrada da camada densa(rede neural)
    Dense(128, activation='relu'),
    Dropout(0.5),
    #saida da camada densa(0-9)
    Dense(10, activation='softmax')
])

# Compila o modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Treina o modelo
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1
)

#salva o modelo
model.save("TRAINED_MODEL.keras")

# Avalia o modelo com os dados de teste
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)
