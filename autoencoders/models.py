from typing import List, Tuple

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from autoencoders.autoencoder import Autoencoder


def get_newsgroup_autoencoder() -> Autoencoder:
    inpt_size = 302
    dense_layers = [128, 302, 128]
    output_size = 302
    autoencoder, encoder = _get_autoencoder(inpt_size, dense_layers, output_size)
    return Autoencoder(autoencoder, encoder)


def _get_autoencoder(inpt_size,
                     dense_layers: List[int],
                     output_size) -> Tuple[Model, Model]:
    inpt = Input(shape=inpt_size)
    x = inpt
    encoder = None
    for idx, layer_size in enumerate(dense_layers):  # type: int, int
        x = Dense(layer_size, activation='relu',)(x)
        if len(dense_layers) // 2 == idx:
            encoder = Model(inpt, x)

    output = Dense(output_size, activation='sigmoid', kernel_initializer='he_uniform')(x)

    autoencoder = Model(inpt, output)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder
