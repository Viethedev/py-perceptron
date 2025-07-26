import numpy as np

# --- Activation Functions ---
class _linear:
    def __init__(self):
        pass
    def get_function(self):
        return lambda x: x
    def get_derivative(self):
        return lambda x: 1
    def get_weights(self, inp, out):
        return basic_uniform_init(inp, out)

class _relu:
    def __init__(self):
        pass
    def get_function(self):
        return lambda x: np.maximum(0, x)
    def get_derivative(self):
        return lambda x: np.where(x > 0, 1, 0)
    def get_weights(self, inp, out):
        return he_init(inp, out)

class _sigmoid:
    def __init__(self):
        pass
    def get_function(self):
        return lambda x: 1 / (1 + np.exp(-x))
    def get_derivative(self):
        return lambda x: x * (1 - x)
    def get_weights(self, inp, out):
        return lecun_init(inp, out)

class _swish:
    def __init__(self):
        pass
    def get_function(self):
        return lambda x: x / (1 + np.exp(-x))
    def get_derivative(self):
        return lambda x: (1 + x + x * np.exp(-x)) / np.square(1 + np.exp(-x))
    def get_weights(self, inp, out):
        return he_init(inp, out)

class _tanh:
    def __init__(self):
        pass
    def get_function(self):
        return lambda x: np.tanh(x)
    def get_derivative(self):
        return lambda x: 1 - np.square(np.tanh(x))
    def get_weights(self, inp, out):
        return xavier_init(inp, out)

class _leaky_relu:
    def __init__(self, a=0.01):
        self.a = a
    def get_function(self):
        return lambda x: np.where(x > 0, x, self.a * x)
    def get_derivative(self):
        return lambda x: np.where(x > 0, 1, self.a)
    def get_weights(self, inp, out):
        return he_init(inp, out)

class _elu:
    def __init__(self, a=1):
        self.a = a
    def get_function(self):
        return lambda x: np.where(x > 0, x, self.a * (np.exp(x) - 1))
    def get_derivative(self):
        return lambda x: np.where(x > 0, 1, self.a * np.exp(x))
    def get_weights(self, inp, out):
        return he_init(inp, out)

class _softmax:
    def __init__(self):
        pass
    def _function(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    def get_function(self):
        return self._function
    def get_derivative(self):
        return lambda y_true, y_pred: y_pred - y_true
    def get_weights(self, inp, out):
        return xavier_init(inp, out)

# --- Activation Functions Instances ---
linear = _linear()
relu = _relu()
sigmoid = _sigmoid()
swish = _swish()
tanh = _tanh()
leaky_relu = lambda a=0.01: _leaky_relu(a)
elu = lambda a=1: _elu(a)
softmax = _softmax()

# --- Loss Functions ---
class binary_crossentropy:
    def _function(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    def _derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])
    def get_function(self):
        return self._function
    def get_derivative(self):
        return self._derivative

class categorical_crossentropy:
    def get_function(self):
        return lambda y_true, y_pred: -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    def get_derivative(self):
        return lambda y_true, y_pred: y_pred - y_true

class mean_squared_error:
    def get_function(self):
        return lambda y_true, y_pred: np.mean((y_true - y_pred)**2)
    def get_derivative(self):
        return lambda y_true, y_pred: 2 * (y_pred - y_true) / y_true.size

# Loss Function Instances
binary_crossentropy = binary_crossentropy()
categorical_crossentropy = categorical_crossentropy()
mean_squared_error = mean_squared_error()

# --- Optimizers ---
class rmsprop:
    def __init__(self, learning_rate, gamma=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.ema = {}
    def update(self, weights, gradients, layer_idx):
        if layer_idx not in self.ema:
            self.ema[layer_idx] = np.zeros_like(gradients)
        g_t = gradients
        ema = self.gamma * self.ema[layer_idx] + (1 - self.gamma) * np.square(g_t)
        self.ema[layer_idx] = ema
        return weights - self.lr / np.sqrt(ema + self.epsilon) * g_t 
           
class adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, weights, gradients, layer_idx):
        if layer_idx not in self.m:
            self.m[layer_idx] = np.zeros_like(gradients)
            self.v[layer_idx] = np.zeros_like(gradients)

        self.t += 1
        self.m[layer_idx] = self.beta1 * self.m[layer_idx] + (1 - self.beta1) * gradients
        self.v[layer_idx] = self.beta2 * self.v[layer_idx] + (1 - self.beta2) * (gradients ** 2)

        m_hat = self.m[layer_idx] / (1 - self.beta1 ** self.t)
        v_hat = self.v[layer_idx] / (1 - self.beta2 ** self.t)

        return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
class sgd:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, weights, gradients, layer_idx):
        return weights - self.lr * gradients

class momentum:
    def __init__(self, learning_rate, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = {}

    def update(self, weights, gradients, layer_idx):
        if layer_idx not in self.v:
            self.v[layer_idx] = np.zeros_like(gradients)
        self.v[layer_idx] = self.momentum * self.v[layer_idx] - self.lr * gradients
        return weights + self.v[layer_idx]

# ---Init methods---
def lecun_init(inp, out):
    std = np.sqrt(1 / inp)
    return np.random.randn(inp, out) * std

def xavier_init(inp, out):
    limit = np.sqrt(6 / (inp + out))
    return np.random.uniform(-limit, limit, (inp, out))

def he_init(inp, out):
    std = np.sqrt(2 / inp)
    return np.random.randn(inp, out) * std

def basic_uniform_init(inp, out):
    return np.random.uniform(-0.1, 0.1, (inp, out))

# --- Utilities ---
def _accuracy(y_true, y_pred):
    if y_true.shape[1] == 1: 
        return np.mean(np.round(y_pred) == y_true)
    else:  
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

# --- Feedforward Neural Network ---
class dense:
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation
        self.activation_function = activation.get_function()
        self.activation_derivative = activation.get_derivative()
        self.weights = None
        self.biases = None

    def build(self, input_dim):
        self.weights = self.activation.get_weights(input_dim, self.units)
        self.biases = np.zeros((1, self.units))

    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.weights) + self.biases
        self.output = self.activation_function(self.z)
        return self.output

    def backward(self, grad_output):
        grad_z = grad_output * self.activation_derivative(self.output)
        grad_w = np.dot(self.input.T, grad_z)
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input, grad_w, grad_b
    
class XFNN:
    """
    Accelerated Feedforward Neural Network (XFNN)

    A fast fully-connected neural network with customizable layers,
    activations, loss functions, optimizers, and training features.
    """
    
    def __init__(self, input_size:int):
        """
        Initialize the network with an input size.

        Args:
            input_size (int): Number of input features.
        """
        
        self.input_size = input_size
        self.layers = []
        self.activation = []
        np.random.seed(42)

    def add_layer(self, layer:dense):
        """
        Add a hidden or output layer to the model.

        Args:
            size (int): Number of neurons in the layer.
            activation (object): Activation function class instance.
        """
        
        self.layers.append(layer)

    def build(self):
        """
        Initialize weights, biases, and activation functions
        after all layers have been added.
        """
        
        input_dim = self.input_size
        for layer in self.layers:
            layer.build(input_dim)
            input_dim = layer.units
            
    def _forward(self, X:np.ndarray):
        """
        Perform a forward pass through the network.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the final layer.
        """
        
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def _backward(self, X:np.ndarray, Y:np.ndarray):
        """
        Perform backward pass and compute gradients.

        Args:
            X (np.ndarray): Input batch.
            Y (np.ndarray): True labels.
        """
        
        output = self._forward(X)
        if isinstance(self.layers[-1].activation, _softmax) and isinstance(self.loss, categorical_crossentropy):
            grad = output - Y
        else:
            grad = self.loss_derivative(Y, output) * self.layers[-1].activation_derivative(output)

        grads_w, grads_b = [], []

        for layer in reversed(self.layers):
            grad, dw, db = layer.backward(grad)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

        self._update_parameters(grads_w, grads_b)

    def _update_parameters(self, grads_w:list, grads_b:list):
        """
        Apply parameter updates using the optimizer.

        Args:
            gradients_w (list): Weight gradients.
            gradients_b (list): Bias gradients.
        """
        
        for i, layer in enumerate(self.layers):
            layer.weights = self.optimizer.update(layer.weights, grads_w[i], f"w{i}")
            layer.biases = self.optimizer.update(layer.biases, grads_b[i], f"b{i}")
            
    def train(self, X_train:np.ndarray, Y_train:np.ndarray, epochs=500, X_val=None, Y_val=None, **kwargs):
        """
        Train the model on the given data.

        Args:
            X_train (np.ndarray): Training input.
            Y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation input.
            Y_val (np.ndarray): Validation labels.
            loss (object): Loss function class.
            epochs (int): Maximum training epochs.
            batch_size (int): Size of training batches.
            learning_rate (float): Learning rate.
            optimizer (class): Optimizer class.
            patience (int): Early stopping patience.
            verbose (bool): Whether to print progress.
            plot (bool): Whether to plot training progress.
            save_best (bool): Save best model weights.
            val_split (float): % of train data to use as validation if val not provided.
        """
        patience=kwargs.get('patience', float('inf'))
        verbose=kwargs.get('verbose', True)
        plot=kwargs.get('plot', False)
        save_best=kwargs.get('save_best', True)
        val_split=kwargs.get('val_split', 0.0)
        
        if X_val is None and val_split > 0:
            split_index = int(len(X_train) * (1 - val_split))
            X_train, X_val = X_train[:split_index], X_train[split_index:]
            Y_train, Y_val = Y_train[:split_index], Y_train[split_index:]

        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []

        self.loss = kwargs.get('loss', mean_squared_error)
        self.loss_fn = self.loss.get_function()
        self.loss_derivative = self.loss.get_derivative()
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.optimizer = kwargs.get('optimizer', sgd)(self.learning_rate)
        batch_size = kwargs.get('batch_size', 64)
        
        num_samples = len(X_train)
        wait = 0

        best_val_loss = float('inf')
        self.best_weights = None

        if plot:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            fig.suptitle("Training Progress")
            ax1.set_ylabel("Loss")
            ax2.set_ylabel("Accuracy")
            ax2.set_xlabel("Epochs")
            loss_line, = ax1.plot([], [], 'r-', label="Train Loss")
            val_loss_line, = ax1.plot([], [], 'g--', label="Val Loss")
            acc_line, = ax2.plot([], [], 'b-', label="Train Acc")
            val_acc_line, = ax2.plot([], [], 'k--', label="Val Acc")
            ax1.legend(); ax2.legend()

            def update_plot():
                x_range = range(len(self.loss_history))
                loss_line.set_data(x_range, self.loss_history)
                val_loss_line.set_data(x_range, self.val_loss_history)
                acc_line.set_data(x_range, self.accuracy_history)
                val_acc_line.set_data(x_range, self.val_accuracy_history)
                ax1.relim(); ax1.autoscale_view()
                ax2.relim(); ax2.autoscale_view()
                plt.draw(); plt.pause(0.001)

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            epoch_losses = []
            epoch_accuracies = []

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                xb, yb = X_shuffled[start:end], Y_shuffled[start:end]

                y_pred = self._forward(xb)
                self._backward(xb, yb)

                loss = self.loss_fn(yb, y_pred)
                acc = _accuracy(yb, y_pred)

                epoch_losses.append(loss)
                epoch_accuracies.append(acc)

            train_loss = np.mean(epoch_losses)
            train_acc = np.mean(epoch_accuracies)
            self.loss_history.append(train_loss)
            self.accuracy_history.append(train_acc)

            if X_val is not None and Y_val is not None:
                val_pred = self._forward(X_val)
                val_loss = self.loss_fn(Y_val, val_pred)
                val_acc = _accuracy(Y_val, val_pred)
            else:
                val_loss = train_loss
                val_acc = train_acc

            self.val_loss_history.append(val_loss)
            self.val_accuracy_history.append(val_acc)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"[Epoch {epoch+1}] "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if plot and (epoch + 1) % 10 == 0:
                update_plot()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                if save_best:
                    self.best_weights = self.get_weights()
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        if save_best and self.best_weights is not None:
            self.set_weights(self.best_weights)

        if plot:
            plt.ioff()
            plt.show()
            
    def get_weights(self):
        """
        Return a copy of current weights and biases.

        Returns:
            dict: Containing 'weights' and 'biases'.
        """
        
        return {
            'weights': [layer.weights.copy() for layer in self.layers],
            'biases': [layer.biases.copy() for layer in self.layers]
            }
    
    def set_weights(self, weights_dict):
        """
        Load a saved weight dictionary into the model.

        Args:
            weights_dict (dict): Dictionary of weights and biases.
        """
        
        for layer, w, b in zip(self.layers, weights_dict['weights'], weights_dict['biases']):
            layer.weights = w.copy()
            layer.biases = b.copy()
        
    def evaluate(self, X:np.ndarray, Y:np.ndarray):
        """
        Evaluate the model on given data.

        Args:
            X (np.ndarray): Input data.
            Y (np.ndarray): Ground truth labels.

        Returns:
            tuple: Loss and accuracy.
        """
        
        y_pred = self._forward(X)
        loss = self.loss_fn(Y, y_pred)
        acc = _accuracy(Y, y_pred)
        print(f"Eval Loss: {loss:.4f} - Accuracy: {acc:.4f}")
        return loss, acc

    def predict(self, X:np.ndarray, round_output=True):
        """
        Predict outputs for new input data.

        Args:
            X (np.ndarray): Input data.
            round_output (bool): Round predictions for classification.

        Returns:
            np.ndarray: Predictions.
        """
        
        pred = self._forward(X)
        return np.round(pred) if round_output else pred

    def summary(self):
        """
        Print a summary of the model's architecture.
        """
        
        print("Model Summary:")
        print(f" Input size: {self.input_size}")
        for i, layer in enumerate(self.layers):
            print(f" Layer {i+1}: {layer.weights.shape[0]} → {layer.weights.shape[1]}  | Activation: {layer.activation.__class__.__name__}")
        print(f" Output size: {self.layers[-1].weights.shape[1]}")

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
    model.train(X, Y, loss=binary_crossentropy, epochs=600, optimizer=adam, plot=False)
    end = time.time()
    model.evaluate(X, Y)
    model.summary()
    print(f"Training Time: {end - start:.2f} seconds")
