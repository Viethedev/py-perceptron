# Py-perceptron
Py-perceptron is a lightweight, efficient, and modular feedforward neural network library built with NumPy. Designed for both learning and experimentation, it provides a clear and intuitive interface for constructing, training, and evaluating neural networks.

# Key Features
Modular Architecture: Easily customize layers and activation functions.

Pure NumPy Implementation: No external dependencies required.

Visual Support: With matplotlib 3.10.0 or higher

Clear API: Designed for educational purposes and rapid prototyping.

Flexible Training: Supports various optimization techniques and loss functions.

# Uses

## Models
There are two models based on two different technique, each one has their unique features and implementations, so try both of them!

### XFNN (Accelerated Feedforward Neural Network)
Fast FNN model with "function-based" logic, faster than modular ones almostly.

### PyIntelligence's Feedforward Neural Network
PyIntelligence's Modular FNN models which offer flexible object-oriented design and advanced training features for deep learning experimentation.

## Examples

### XFNN (Accelerated Feedforward Neural Network)
```bash
from perceptron import XFNN

if __name__ ==  "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ])
    Y = np.array([
        [0],
        [1],
        [1],
        [0]
        ])
    model = XFNN(input_size=2)
    model.add_layer(dense(2, activation=tanh))
    model.add_layer(dense(1, activation=sigmoid))
    model.build()
    import time
    start = time.time()
    model.train(X, Y, loss=binary_crossentropy, epochs=600, optimizer=adam, plot=True)
    end = time.time()
    model.evaluate(X, Y)
    model.summary()
    print(f"Training Time: {end - start:.2f} seconds")
```
### PyIntelligence's Feedforward Neural Network
```bash
from perceptron import Feedforward
if __name__ == "__main__":
    import time

    start = time.time()

    model = Feedforward()

    model.layers.input = 2
    model.layers.add_Tanh(2)
    model.layers.add_Sigmoid(1)

    model.loss.use_BinaryCE()
    model.optimizer.use_Adam()

    model.build()
    x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.uint8)
    y = np.array([[0],[1],[1],[0]], dtype=np.uint8)
    prepare = time.time()

    model.train(x, y, epochs=600, plot=False)
    train = time.time()

    print('Predicts:\n', model.predict(x))
    predict = time.time()
    model.summary()

    print('Preparing time: ', prepare - start)
    print('Training time: ', train - prepare)
    print('Predicting time: ', predict - train)
```
# Install
```bash
pip install perceptron
```

# External Links
Github: https://github.com/Viethedev/py-perceptron

# Other projects
PyIntelligence: https://pypi.org/project/PyIntelligence




