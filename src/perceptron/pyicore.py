import numpy as np

def lecun_init(inp, out):
    return np.random.randn(inp, out) * np.sqrt(1 / inp)

def xavier_init(inp, out):
    limit = np.sqrt(6 / (inp + out))
    return np.random.uniform(-limit, limit, (inp, out))

def he_init(inp, out):
    return np.random.randn(inp, out) * np.sqrt(2 / inp)

def basic_uniform_init(inp, out):
    return np.random.uniform(-0.1, 0.1, (inp, out))

class Linear:
    def __init__(self, units):
        self.units = units
        
    def build(self, input_dim, optimizer: object):
        self.weights = basic_uniform_init(input_dim, self.units)
        self.biases = np.zeros((1, self.units))
        self.w_opt_cache = optimizer.setup(self.weights.shape)
        self.b_opt_cache = optimizer.setup(self.biases.shape)
        self.update = optimizer.update

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        return self.z

    def backward(self, grad_z):
        grad_w = self.input.T @ grad_z
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.weights.T
        self.weights, self.w_opt_cache = self.update(self.weights, grad_w, self.w_opt_cache)
        self.biases, self.b_opt_cache = self.update(self.biases, grad_b, self.b_opt_cache)
        return grad_input
    
class ReLU:
    def __init__(self, units):
        self.units = units

    def build(self, input_dim, optimizer: object):
        self.weights = he_init(input_dim, self.units)
        self.biases = np.zeros((1, self.units))
        self.w_opt_cache = optimizer.setup(self.weights.shape)
        self.b_opt_cache = optimizer.setup(self.biases.shape)
        self.update = optimizer.update

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        self.output = np.maximum(0, self.z)
        return self.output

    def backward(self, grad_output):
        grad_z = grad_output * (self.output > 0)
        grad_w = self.input.T @ grad_z
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.weights.T
        self.weights, self.w_opt_cache = self.update(self.weights, grad_w, self.w_opt_cache)
        self.biases, self.b_opt_cache = self.update(self.biases, grad_b, self.b_opt_cache)
        return grad_input
        
class Sigmoid:
    def __init__(self, units):
        self.units = units

    def build(self, input_dim, optimizer: object):
        self.weights = lecun_init(input_dim, self.units)
        self.biases = np.zeros((1, self.units))
        self.w_opt_cache = optimizer.setup(self.weights.shape)
        self.b_opt_cache = optimizer.setup(self.biases.shape)
        self.update = optimizer.update

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        self.output = 1 / (1 + np.exp(-self.z))
        return self.output

    def backward(self, grad_output):
        grad_z = grad_output * self.output * (1 - self.output)
        grad_w = self.input.T @ grad_z
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.weights.T
        self.weights, self.w_opt_cache = self.update(self.weights, grad_w, self.w_opt_cache)
        self.biases, self.b_opt_cache = self.update(self.biases, grad_b, self.b_opt_cache)
        return grad_input

class Swish:
    def __init__(self, units):
        self.units = units

    def build(self, input_dim, optimizer: object):
        self.weights = he_init(input_dim, self.units)
        self.biases = np.zeros((1, self.units))
        self.w_opt_cache = optimizer.setup(self.weights.shape)
        self.b_opt_cache = optimizer.setup(self.biases.shape)
        self.update = optimizer.update

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        self.sigmoid_z = 1 / (1 + np.exp(-self.z))
        self.output = self.z * self.sigmoid_z
        return self.output

    def backward(self, grad_output):
        grad_z = grad_output * (self.sigmoid_z + self.output * (1 - self.sigmoid_z))
        grad_w = self.input.T @ grad_z
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.weights.T
        self.weights, self.w_opt_cache = self.update(self.weights, grad_w, self.w_opt_cache)
        self.biases, self.b_opt_cache = self.update(self.biases, grad_b, self.b_opt_cache)
        return grad_input

class LeakyReLU:
    def __init__(self, units, a=0.01):
        self.units = units
        if a < 0:
            print("Leaky ReLU parameter 'a' must be positive!\nSet a = 0.01")
            self.a = 0.01
        else:
            self.a = a

    def build(self, input_dim, optimizer: object):
        self.weights = he_init(input_dim, self.units)
        self.biases = np.zeros((1, self.units))
        self.w_opt_cache = optimizer.setup(self.weights.shape)
        self.b_opt_cache = optimizer.setup(self.biases.shape)
        self.update = optimizer.update

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        self.mask = (self.z > 0) + (self.z <= 0) * self.a
        self.output = self.z * self.mask
        return self.output

    def backward(self, grad_output):
        grad_z = grad_output * self.mask
        grad_w = self.input.T @ grad_z
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.weights.T
        self.weights, self.w_opt_cache = self.update(self.weights, grad_w, self.w_opt_cache)
        self.biases, self.b_opt_cache = self.update(self.biases, grad_b, self.b_opt_cache)
        return grad_input

class Tanh:
    def __init__(self, units):
        self.units = units

    def build(self, input_dim, optimizer: object):
        self.weights = xavier_init(input_dim, self.units)
        self.biases = np.zeros((1, self.units))
        self.w_opt_cache = optimizer.setup(self.weights.shape)
        self.b_opt_cache = optimizer.setup(self.biases.shape)
        self.update = optimizer.update

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        self.output = np.tanh(self.z) 
        return self.output

    def backward(self, grad_output):
        grad_z = grad_output * (1 - self.output ** 2) 
        grad_w = self.input.T @ grad_z
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.weights.T
        self.weights, self.w_opt_cache = self.update(self.weights, grad_w, self.w_opt_cache)
        self.biases, self.b_opt_cache = self.update(self.biases, grad_b, self.b_opt_cache)
        return grad_input

class ELU:
    def __init__(self, units, a=1):
        self.units = units
        if a < 0:
            print("ELU parameter 'a' must be positive!\nSet a = 1")
            self.a = 1
        else:
            self.a = a

    def build(self, input_dim, optimizer: object):
        self.weights = he_init(input_dim, self.units)
        self.biases = np.zeros((1, self.units))
        self.w_opt_cache = optimizer.setup(self.weights.shape)
        self.b_opt_cache = optimizer.setup(self.biases.shape)
        self.update = optimizer.update

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        self.output = self.z * (self.z > 0) + self.a * (np.exp(self.z) - 1) * (self.z <= 0)
        return self.output

    def backward(self, grad_output):
        grad_z = grad_output * ((self.z > 0) + (self.z <= 0) * self.a * np.exp(self.z))
        grad_w = self.input.T @ grad_z
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.weights.T
        self.weights, self.w_opt_cache = self.update(self.weights, grad_w, self.w_opt_cache)
        self.biases, self.b_opt_cache = self.update(self.biases, grad_b, self.b_opt_cache)
        return grad_input

class ParametricReLU:
    def __init__(self, units, alpha='flexible'):    
        self.units = units
        if alpha == 'flexible':
            self.a = np.ones((1, self.units)) * 0.01
        elif isinstance(alpha, (int, float)):
            self.a = alpha
        else:
            print('alpha must be a number!\nSet alpha = 0.01.')
            self.a = 0.01
            
    def build(self, input_dim, optimizer: object):
        self.weights = he_init(input_dim, self.units)
        self.biases = np.zeros((1, self.units))
        self.w_opt_cache = optimizer.setup(self.weights.shape)
        self.b_opt_cache = optimizer.setup(self.biases.shape)
        self.a_opt_cache = optimizer.setup(self.a.shape)
        self.update = optimizer.update

    def forward(self, x):
        self.a = np.maximum(1e-8, self.a)
        self.input = x
        self.z = x @ self.weights + self.biases
        self.output = np.maximum(self.a * self.z, self.z)
        return self.output

    def backward(self, grad_output):
        grad_z = grad_output * ((self.z > 0) + (self.z <= 0) * self.a)
        grad_w = self.input.T @ grad_z
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.weights.T
        grad_a = np.sum(grad_output * (self.z <= 0) * self.z, axis=0, keepdims=True)
        self.weights, self.w_opt_cache = self.update(self.weights, grad_w, self.w_opt_cache)
        self.biases, self.b_opt_cache = self.update(self.biases, grad_b, self.b_opt_cache)
        self.a, self.a_opt_cache = self.update(self.a, grad_a, self.a_opt_cache)
        return grad_input

class Softmax:
    def __init__(self, units):
        self.units = units

    def build(self, input_dim, optimizer: object):
        self.weights = xavier_init(input_dim, self.units)
        self.biases = np.zeros((1, self.units))
        self.w_opt_cache = optimizer.setup(self.weights.shape)
        self.b_opt_cache = optimizer.setup(self.biases.shape)
        self.update = optimizer.update

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        exps = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        grad_z = grad_output
        grad_w = self.input.T @ grad_z
        grad_b = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.weights.T
        self.weights, self.w_opt_cache = self.update(self.weights, grad_w, self.w_opt_cache)
        self.biases, self.b_opt_cache = self.update(self.biases, grad_b, self.b_opt_cache)
        return grad_input

class BinaryCrossEntropy:
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def forward_w_logits(self, y_true, z):
        return np.mean(np.maximum(0, z) - z * y_true + np.log(1 + np.exp(-np.abs(z))))

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])

    def backward_w_logits(self, y_true, z):
        return (1 / (1 + np.exp(-z)) - y_true) / y_true.shape[0]

class CategoricalCrossEntropy:
    def __init__(self):
        pass
    
    def forward(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
    
    def backward(self, y_true, y_pred):
        return y_pred - y_true

class MeanSquaredError:
    def __init__(self):
        pass
    
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class MeanAbsoluteError:
    def __init__(self):
        pass
    
    def forward(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def backward(self, y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.size

class SparseCrossEntropy:
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        correct_log_probs = -np.log(y_pred_clipped[np.arange(y_pred.shape[0]), y_true])
        return np.mean(correct_log_probs)

    def backward(self, y_true, y_pred):
        grad = np.clip(y_pred, 1e-15, 1 - 1e-15)
        grad[np.arange(y_pred.shape[0]), y_true] -= 1
        return grad / y_pred.shape[0]

class RMSProp:
    def __init__(self, learning_rate, gamma=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
    def setup(self, param_shape):        
        return np.zeros(param_shape)
        
    def update(self, param, grad, ema):
        g_t = grad
        ema = self.gamma * ema + (1 - self.gamma) * np.square(g_t)
        return param - self.lr / np.sqrt(ema + self.epsilon) * g_t, ema

class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def setup(self, param_shape):        
        return {'m': np.zeros(param_shape), 'v': np.zeros(param_shape), 't': 0}

    def update(self, param, grad, cache):
        m = cache['m']
        v = cache['v']
        t = cache['t'] + 1
        
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        cache['m'] = m
        cache['v'] = v
        cache['t'] = t

        return param, cache
    
class Vanilla:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def setup(self, param_shape):
        return {}

    def update(self, param, grad):
        return param - self.lr * grad, {}

class Momentum:
    def __init__(self, learning_rate, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum

    def setup(self, param_shape):
        return np.zeros(param_shape)

    def update(self, param, grad, v):
        v = self.momentum * v - self.lr * grad
        return param + v, v

class layers:
    def __init__(self):
        self.hidden_layers = []

    def set_Input(self, size):
        self.input = size

    def add_Linear(self, units):
        self.hidden_layers.append(Linear(units))
        
    def add_ReLU(self, units):
        self.hidden_layers.append(ReLU(units))

    def add_Sigmoid(self, units):
        self.hidden_layers.append(Sigmoid(units))
        
    def add_Swish(self, units):
        self.hidden_layers.append(Swish(units))

    def add_LeakyReLU(self, units, a=0.01):
        self.hidden_layers.append(LeakyReLU(units, a))
        
    def add_Tanh(self, units):
        self.hidden_layers.append(Tanh(units))

    def add_ELU(self, units, a=1):
        self.hidden_layers.append(ELU(units, a))
        
    def add_Swish(self, units):
        self.hidden_layers.append(Swish(units))

    def add_PReLU(self, units, a='flexible'):
        """
        Initializes the parametric ReLU layer.

        Args:
            units (int): Number of units in the layer.
            alpha (int, float, or str): Alpha parameter value or mode. 
                Use 'flexible' for a learnable alpha parameter per unit.

        Notes:
            The 'flexible' mode allows each unit to have its own learnable alpha parameter,
            which requires more computational resources.
        """
        self.hidden_layers.append(ParametricReLU(units, a))
        
    def add_Softmax(self, units):
        self.hidden_layers.append(Softmax(units))

class loss:
    def __init__(self):
        self.in_use = None

    def use_BinaryCE(self):
        self.in_use = BinaryCrossEntropy()

    def use_CategoricalCE(self):
        self.in_use = CategoricalCrossEntropy()

    def use_SparseCE(self):
        self.in_use = SparseCrossEntropy()

    def use_MSE(self):
        self.in_use = MeanSquaredError()

    def use_MAE(self):
        self.in_use = MeanAbsoluteError()

class optimizer:
    def __init__(self):
        self.in_use = None

    def use_RMSProp(self, learning_rate=0.01, gamma=0.9, epsilon=1e-8):
        self.in_use = RMSProp(learning_rate, gamma, epsilon)

    def use_Adam(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.in_use = Adam(learning_rate, beta1, beta2, epsilon)

    def use_Momentum(self, learning_rate=0.01, momentum=0.9):
        self.in_use = Momentum(learning_rate, momentum)

    def use_Vanilla(self, learning_rate=0.01):
        self.in_use = Vanilla(learning_rate)
    
class Feedforward:
    """
    Feedforward Neural Network (FNN)

    A modular fully-connected neural network with customizable layers,
    activations, loss functions, optimizers, and training features.

    """
    def __init__(self) -> None:
        """
        Initialize model.
        """  
        self.layers = layers()
        np.random.seed(42)
        self.loss = loss()
        self.optimizer = optimizer()
 
    def build(self) -> None:    
        """
        Build the model architecture based on modifications.
        """
        self.opt = self.optimizer.in_use
        self.denses = self.layers.hidden_layers
        input_dim = self.layers.input
        for layer in self.denses:
            layer.build(input_dim, self.opt)
            input_dim = layer.units

    def predict(self, x:np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the final layer.
        """
        
        output = x
        for layer in self.denses:
            output = layer.forward(output)
        return output

    def train(self, X_train:np.ndarray, Y_train:np.ndarray, epochs=100, 
              X_val=None, Y_val=None, val_split=0.0, batch_size=None, 
              patience=float('inf'), verbose=True, save_best=False,
              plot=False) -> None:        
        """
        Train the model on the given data.

        Args:
            X_train (np.ndarray): Training input.
            Y_train (np.ndarray): Training labels.
            epochs (int): Maximum training epochs.
            X_val (np.ndarray): Validation input.
            Y_val (np.ndarray): Validation labels.
            val_split (float): % of train data to use as validation if val not provided.
            batch_size (int): Size of training batches.
            patience (int): Early stopping patience.
            verbose (bool): Whether to print progress.
            save_best (bool): Save best model weights.
            plot (bool): Whether to plot training progress.
        """
        if X_val is not None and Y_val is not None:
            validation_is_on = True
        elif X_val is None and Y_val is None and val_split == 0:
            validation_is_on = False
        elif val_split > 0:
            split_index = int(len(X_train) * (1 - val_split))
            X_train, X_val = X_train[:split_index], X_train[split_index:]
            Y_train, Y_val = Y_train[:split_index], Y_train[split_index:]
            validation_is_on = True
        else:
            validation_is_on = False

        num_samples = len(X_train)       
        
        loss_fd = self.loss.in_use.forward
        loss_bd = self.loss.in_use.backward
        
        if batch_size == None:
            batch_size = num_samples
            
        early_stopping = patience < float('inf')
        
        if save_best:
            best_val_loss = float('inf')
            self.best_weights = None
            
        save_best_or_early_stopping = save_best or early_stopping
        
        if save_best_or_early_stopping:
            wait = 0
        
        num_batches = int(num_samples / batch_size)
        
        if plot:       
            import matplotlib.pyplot as plt
            self.loss_history = np.zeros(epochs)
            plt.ion()
            if validation_is_on: self.val_loss_history = np.zeros(epochs)
            fig, ax1 = plt.subplots(figsize=(8, 4))
            fig.suptitle("Training Progress")
            ax1.set_ylabel("Loss")
            ax1.set_xlabel("Epochs")
            loss_line, = ax1.plot([], [], 'r-', label="Train Loss")
            if validation_is_on: val_loss_line, = ax1.plot([], [], 'g--', label="Val Loss") 
            ax1.legend()             
            
        for epoch in range(1, epochs+1):
            indices = np.random.permutation(num_samples)
            
            X_shuffled = np.array_split(X_train[indices], num_batches, axis=0)
            Y_shuffled = np.array_split(Y_train[indices], num_batches, axis=0)

            total_loss = 0.0

            for xb, yb in zip(X_shuffled, Y_shuffled):            
                output = xb
                for layer in self.denses:
                    output = layer.forward(output)               
                grad = loss_bd(yb, output)
            
                for layer in reversed(self.denses):
                    grad = layer.backward(grad)
                    
                total_loss += loss_fd(yb, output)
                
            train_loss = total_loss / num_batches
            
            if validation_is_on:
                val_output = X_val
                for layer in self.denses:
                    val_output = layer.forward(val_output)
                val_loss = loss_fd(Y_val, val_output) 
            else:
                val_loss = train_loss.copy()                  

            if verbose and epoch % 100 == 0:
                val_log = f" | Val Loss: {val_loss:.4f}" if validation_is_on else ""
                print(f"[Epoch {epoch}/{epochs}] Train Loss: {train_loss:.4f}{val_log}")

            if save_best_or_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                    if save_best:
                        self.best_weights = self.get_weights()
                elif early_stopping:
                    wait += 1
                    if wait >= patience:
                        if verbose: print(f"Early stopping at epoch {epoch}")
                        break
            if plot:
                self.loss_history[epoch-1] = train_loss
                if validation_is_on: self.val_loss_history[epoch-1] = val_loss
                if epoch % 10 == 0:
                    x_range = range(epoch)
                    loss_line.set_data(x_range, self.loss_history[:epoch])
                    if validation_is_on: val_loss_line.set_data(x_range, self.val_loss_history[:epoch])
                    ax1.relim(); ax1.autoscale_view()
                    plt.draw(); plt.pause(0.001)
        if plot:
            plt.ioff()
            plt.show()

        if save_best and self.best_weights is not None:
            self.set_weights(self.best_weights)

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
        
    def summary(self):
        print(f"===== ===== ===== Summary ===== ===== =====")
        print("Layer		Units		Activation")
        print(f"Input		{self.layers.input}")
        for i, layer in enumerate(self.denses):
            if i+1 < len(self.denses):
                print(f"Hidden {i+1}{' ' * (16 - len(f'Hidden {i+1}'))}{layer.weights.shape[0]}{' ' * (16 - len(str(layer.weights.shape[0])))}{layer.__class__.__name__}")
            else:
                print(f"Output{' ' * 10}{layer.weights.shape[0]}{' ' * (16 - len(str(layer.weights.shape[0])))}{layer.__class__.__name__}")	

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
