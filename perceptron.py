import json
import numpy as np


def sigmoid(x, beta=1.0):
    return 1.0 / (1.0 + np.exp(-beta * x))


def sigmoid_derivative(a, beta=1.0):
    # a — значение sigmoid
    return beta * a * (1.0 - a)


def tanh(x, beta=1.0):
    return np.tanh(beta * x)


def tanh_derivative(a, beta=1.0):
    # a — значение tanh
    return beta * (1.0 - a ** 2)


def relu(x, beta=1.0):
    return np.maximum(0.0, beta * x)


def relu_derivative(a, beta=1.0):
    # a — уже значение relu(beta * x)
    return np.where(a > 0.0, beta, 0.0)


class MLP:
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        lr=0.1,
        activation="sigmoid",
        beta=1.0,              # чувствительность
        inertia=0.0,           # инерционность (momentum)
        correction="gradient", # gradient / momentum
        threshold=0.5          # порог уверенности
    ):
        self.lr = float(lr)
        self.beta = float(beta)
        self.inertia = float(inertia)
        self.correction = correction
        self.threshold = float(threshold)
        self.activation_name = activation

        self.layers = [input_size] + list(hidden_layers) + [output_size]

        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == "tanh":
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Unsupported activation. Use sigmoid, tanh, relu.")

        # Инициализация весов
        self.weights = []
        self.velocities = []

        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, (fan_in, fan_out))
            self.weights.append(w)
            self.velocities.append(np.zeros_like(w))

    def forward(self, X):
        self.activations = [X]

        # Скрытые слои
        for W in self.weights[:-1]:
            X = self.activation(np.dot(X, W), beta=self.beta)
            self.activations.append(X)

        # Выходной слой sigmoid для классификации
        out = sigmoid(np.dot(X, self.weights[-1]), beta=self.beta)
        self.activations.append(out)
        return out

    def train(self, X, y, epochs=1000, verbose=True):
        errors = []
        n = X.shape[0]

        for epoch in range(epochs):
            output = self.forward(X)
            error = y - output
            errors.append(float(np.mean(np.abs(error))))

            # Ошибка на выходе
            delta = error * sigmoid_derivative(output, beta=self.beta)

            # Обратное распространение
            for i in reversed(range(len(self.weights))):
                a_prev = self.activations[i]
                W_current = self.weights[i].copy()

                grad = a_prev.T.dot(delta) / n

                if self.correction == "momentum":
                    self.velocities[i] = self.inertia * self.velocities[i] + self.lr * grad
                    self.weights[i] += self.velocities[i]
                else:
                    self.weights[i] += self.lr * grad

                if i > 0:
                    a_hidden = self.activations[i]
                    delta = delta.dot(W_current.T) * self.activation_derivative(a_hidden, beta=self.beta)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Error: {errors[-1]:.6f}")

        return errors

    def predict(self, X):
        return self.forward(X)

    def predict_class(self, X):
        probs = self.predict(X)
        cls = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)

        # если уверенность ниже порога считаем ответ неопределённым
        cls = np.where(conf >= self.threshold, cls, -1)
        return cls, conf, probs

    def save(self, path="model.json"):
        data = {
            "layers": self.layers,
            "lr": self.lr,
            "beta": self.beta,
            "inertia": self.inertia,
            "correction": self.correction,
            "threshold": self.threshold,
            "activation": self.activation_name,
            "weights": [w.tolist() for w in self.weights],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path="model.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        layers = data["layers"]
        obj = cls(
            input_size=layers[0],
            hidden_layers=layers[1:-1],
            output_size=layers[-1],
            lr=data["lr"],
            activation=data["activation"],
            beta=data["beta"],
            inertia=data["inertia"],
            correction=data["correction"],
            threshold=data["threshold"],
        )
        obj.weights = [np.array(w) for w in data["weights"]]
        obj.velocities = [np.zeros_like(w) for w in obj.weights]
        return obj