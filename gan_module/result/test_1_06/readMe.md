# Different generator model comparison

## model 1 

- CNN signal interference generator

- Comparing noise generator in MLP model with CNN model

## Test 2

- Using more complex structure of generator

```python
# mlp model

def mlp_generator(blockSize):
    model = tf.keras.Sequential()
    model.add(layers.Reshape((blockSize,3), input_shape=(blockSize, 3)))
    model.add(layers.Dense(16,  activation="linear"))
    model.add(layers.Dense(32, activation="linear"))
    model.add(layers.Dense(64, activation="linear"))
    model.add(layers.Dense(32, activation="linear"))
    model.add(layers.Dense(16, activation="linear"))
    model.add(layers.Dense(2))
    model.add(layers.Reshape((blockSize,2,1)))
    return model

```

```python
# cnn model

def make_generator(blockSize):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (1, 3), strides=(1,3), activation="linear",
                            input_shape=(blockSize, 3, 1)))
    model.add(layers.Conv2D(32, (1,16), activation="linear", padding='same'))
    model.add(layers.Conv2D(16, (1,32), activation="linear", padding='same'))
    model.add(layers.Reshape((blockSize, 16, 1)))
    model.add(layers.AveragePooling2D((1,8)))
    model.add(layers.Dense(1))
    return model

```