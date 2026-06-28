from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input


def nvidia_model():
    model = Sequential([
        Input(shape=(76, 320, 3)),

        # Bloque convolucional — extracción de características
        Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),

        Flatten(),

        # Bloque denso — regresión del ángulo de dirección
        Dense(100, activation='elu'),
        Dropout(0.5),
        Dense(50,  activation='elu'),
        Dropout(0.5),
        Dense(10,  activation='elu'),
        Dense(1),
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model
