import numpy as np
import keras

class DQNModel:
    def __init__(self):
        self.kernel = self.create_model()

    def create_model(self):
        model = keras.Sequential()

        model.add(keras.layers.Conv2D(8, (3, 3), input_shape=(3, 3, 1)))
        model.add(keras.layers.Activation('relu'))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(9, activation='linear'))
        model.compile(loss="mse", optimizer=keras.optimizers.RMSprop(0.001), metrics=['accuracy'])
        return model

    def get_action_space_size(self):
        return 9

    def get_top_action(self, inputs):
        outputs = self.predict(inputs)
        return np.argmax(outputs[0])

    def save(self, path):
        self.kernel.save(path)

    def predict(self, inputs):
        inputs = inputs.reshape((-1, 3, 3, 1))
        return self.kernel.predict(inputs)

    def fit(self, *args, **kwargs):
        return self.kernel.fit(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.kernel.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.kernel.set_weights(*args, **kwargs)

