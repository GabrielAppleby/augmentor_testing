import numpy as np
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping


class Autoencoder:
    def __init__(self,
                 autoencoder_impl: Model,
                 encoder_impl: Model):
        super().__init__()
        self._autoencoder_impl = autoencoder_impl
        self._encoder_impl = encoder_impl

    def fit(self, x: np.array):
        return self._autoencoder_impl.fit(x=x,
                                          y=x,
                                          verbose=0,
                                          batch_size=32,
                                          validation_split=.1,
                                          epochs=50)
                                          # callbacks=[EarlyStopping(patience=10,
                                          #                          restore_best_weights=True)])

    def encode(self, x: np.array):
        return self._encoder_impl.predict(x)
