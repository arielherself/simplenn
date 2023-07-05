from copy import deepcopy
from typing import Iterable, List
from abc import abstractmethod, ABC
import numpy as np
from numpy.random import rand
import progressbar


def round_u(x: float) -> int:
    return int(x + 0.5)


def reversed_enumerate(x: Iterable):
    x = list(x)
    i = len(x)
    while i > 0:
        i -= 1
        yield i, x[i]


class ActivationFuncGenerator(ABC):
    @abstractmethod
    def call(self, z: float | np.float64 | np.ndarray) -> np.float64 | np.ndarray: pass

    @abstractmethod
    def df_dz(self, z: float | np.float64 | np.ndarray) -> np.float64 | np.ndarray: pass


class LossFuncGenerator(ABC):
    @abstractmethod
    def call(self, p: float | np.float64 | np.ndarray, y: float | np.float64 | np.ndarray) -> np.float64 | np.ndarray:
        pass

    @abstractmethod
    def dl_dp(self, p: float | np.float64 | np.ndarray, y: float | np.float64 | np.ndarray) -> np.float64 | np.ndarray:
        """
        :param p: predicted value
        :param y: target value
        :return: a positive number
        """
        pass


class Sigmoid(ActivationFuncGenerator):
    def call(self, z: float | np.float64 | np.ndarray) -> np.float64 | np.ndarray:
        return 1 / (1 + np.exp(-z))

    def df_dz(self, z: float | np.float64 | np.ndarray) -> np.float64 | np.ndarray:
        if isinstance(z, np.ndarray):
            return np.array([np.exp(-z[i]) / np.power((1 + np.exp(-z[i])), 2) for i in range(z.shape[0])])
        return np.exp(-z) / np.power((1 + np.exp(-z)), 2)


class CrossEntropy(LossFuncGenerator):
    def call(self, p: float | np.float64, y: float | np.float64 | np.ndarray) -> np.float64 | np.ndarray:
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def dl_dp(self, p: float | np.float64, y: float | np.float64 | np.ndarray) -> np.float64 | np.ndarray:
        if isinstance(p, np.ndarray) and isinstance(y, np.ndarray):
            return np.array([np.float64(- y[i] / p[i] + (1 - y[i]) / (1 - p[i])) for i in range(p.shape[0])])
        else:
            return np.float64(- y / p + (1 - y) / (1 - p))

    def __init__(self):
        pass


class Dense:
    def __init__(self, units: int, activation_func: ActivationFuncGenerator, params: int):
        self.units, self.activation_func = units, activation_func
        self.params = params
        self.w = rand(units, params) / 10.0
        self.b = rand(units) / 10.0

    def predict(self, x: np.ndarray, no_activation: bool = False) -> np.ndarray:
        assert x.shape == (self.params,)
        return (np.matmul(self.w, x) + self.b) if no_activation \
            else self.activation_func.call(np.matmul(self.w, x) + self.b)


class Model:
    def __init__(self, units: np.ndarray, input_params: int, activation_func: ActivationFuncGenerator,
                 loss_func: LossFuncGenerator):
        self.layers: list[Dense] = []
        self.input_params = input_params
        self.activation_func = activation_func
        self.loss_func = loss_func
        f = input_params
        for u in units:
            self.layers.append(Dense(u, activation_func, f))
            f = u
        self.total_params = sum([layer.w.size for layer in self.layers])
        self.total_units = sum([layer.w.shape[0] for layer in self.layers])
        if units[-1] != 1:
            print('WARNING: Not a predicting model')

    def get_w_tensor(self):
        return np.array([d.w for d in self.layers])

    def get_w_sum(self):
        return sum([np.sum(layer.w) for layer in self.layers])

    def new_cache(self):
        r = []
        for layer in self.layers:
            n = np.zeros(layer.w.shape)
            r.append(n)
        return np.array(r, dtype=object)

    def predict(self, feature: np.ndarray) -> np.ndarray:
        assert feature.shape == (self.input_params,)
        o = feature
        for layer in self.layers:
            o = layer.predict(o)
        return o

    def predict_verbose(self, feature: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        get every `z` and `a` in the predicting process.
        :param feature: feature
        :return: a tuple comprising two lists `z` and `a` of size [layers, units]
        """
        assert feature.shape == (self.input_params,)
        z = []
        a = []
        o = feature
        for layer in self.layers:
            z.append(layer.predict(o, no_activation=True))
            a.append(layer.predict(o))
            o = layer.predict(o)
        return z, a

    def get_loss(self, x: np.ndarray, y: np.ndarray) -> np.float64:
        l = np.float64(0)
        for i in range(x.shape[0]):
            l += self.loss_func.call(self.predict(x[i])[0], y[i])
        l /= x.shape[0]
        return l

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.float64:
        pbar = progressbar.ProgressBar(
            widgets=['Verifying ', progressbar.Percentage('%(percentage)3d%%'), ' ', progressbar.Bar('#'), ' ',
                     progressbar.Timer(), ' ', progressbar.ETA(), ' '], maxval=100)
        correct, incorrect = 0, 0
        for i in range(y.size):
            pbar.update(min((i + 1) / y.size * 100.0, 100))
            if round_u(float(self.predict(x[i]))) == y[i]:
                correct += 1
            else:
                incorrect += 1
        pbar.finish()
        print(f'\nTotal: {y.size}, Correct: {correct}, Incorrect: {incorrect}')
        print(f'Rate: {correct / y.size * 100.0:.4f}%, Loss: {self.get_loss(x, y)}')
        return self.get_loss(x, y)

    def train(self, x: np.ndarray, y: np.ndarray, epoch_count: int, alpha: float):
        """
        Train the model using given data set.
        :param x: array of size (m, n)
        :param y: array of size (n, )
        :param epoch_count: epoch count
        :param alpha: learning rate
        :return: the model itself
        """
        assert x.shape[1] == self.input_params
        print('WARNING: Start training ...')
        new_layers = deepcopy(self.layers)
        # don't repeatedly calculate prediction, `z` and `a` in each epoch
        for e in range(epoch_count):
            print(f'Epoch {e + 1}/{epoch_count}, loss {self.get_loss(x, y)}')
            pbar = progressbar.ProgressBar(
                widgets=[progressbar.Percentage('%(percentage)3.4f%%'), ' ', progressbar.Bar('#'), ' ',
                         progressbar.Timer(), ' ', progressbar.ETA(), ' '], maxval=100)
            c = 0
            cc = self.total_params * len(x)
            for i in range(len(x)):
                feature, target = x[i], y[i]
                f = len(self.layers) - 1
                z, a = self.predict_verbose(feature)
                prediction = float(self.predict(feature))
                factor = alpha * self.loss_func.dl_dp(prediction, target)
                cache = self.new_cache()
                for l, layer in reversed_enumerate(self.layers):
                    for j in range(layer.units):
                        pbar.update(c / cc * 100.0)
                        for k in range(layer.params):
                            c += 1
                            new_layers[l].w[j, k] -= factor * self.pd(z, a, j, k, l, 0, f, feature, cache)
                self.layers = new_layers
            pbar.finish()
        return self

    def pd(self, z: List[np.ndarray], a: List[np.ndarray], t_unit: int, t_param: int, t_layer: int,
           c_unit: int, c_layer: int, features: np.ndarray, cache: np.ndarray) -> int:
        result = 0
        for i in range(cache[t_layer + 1].shape[0]):
            result += cache[t_layer + 1]
        result = self.activation_func.df_dz(z[c_layer][c_unit])
        if c_layer == t_layer and t_unit == c_unit:
            if c_layer == 0:
                result *= features[t_param]
            else:
                result *= a[c_layer - 1][t_param]
        elif c_layer == t_layer and t_unit != c_unit:
            result = 0
        else:
            total_params = self.layers[c_layer - 1].units
            r = 0
            for i in range(total_params):
                r += self.layers[c_layer].w[c_unit, i] \
                     * self.pd(z, a, t_unit, t_param, t_layer, i, c_layer - 1, features)
            result *= r
        cache[t_layer][t_unit, t_param] = result
        return result

    def train_ng(self, x: np.ndarray, y: np.ndarray, epoch_count: int, alpha: float):
        assert x.shape[1] == self.input_params
        for e in range(epoch_count):
            print(f'\nEpoch {e + 1}/{epoch_count}, loss {self.get_loss(x, y)}')
            pbar = progressbar.ProgressBar(
                widgets=[progressbar.Percentage('%(percentage)3.4f%%'), ' ', progressbar.Bar('#'), ' ',
                         progressbar.Timer(), ' ', progressbar.ETA(), ' '], maxval=100)
            c = 0
            cc = self.total_units * len(x)
            c += 1
            pbar.update(min(c / cc * 100.0, 100))
            for i in range(len(x)):
                dj_da = [[0] * layer.units for layer in self.layers]
                dj_dw = self.new_cache()
                dj_db = [[0] * layer.units for layer in self.layers]
                layers = deepcopy(self.layers)
                feature, target = x[i], y[i]
                z, a = self.predict_verbose(feature)
                prediction = float(self.predict(feature))
                dj_da[-1][0] = self.loss_func.dl_dp(prediction, target)
                tp = self.loss_func.dl_dp(prediction, target)
                for m in range(layers[-1].params):
                    dj_dw[-1][0, m] = tp * self.activation_func.df_dz(z[-1][0]) * a[-2][m]
                    layers[-1].w[0, m] -= alpha * dj_dw[-1][0, m]
                dj_db[-1][0] = tp * self.activation_func.df_dz(z[-1][0])
                layers[-1].b[0] -= alpha * dj_db[-1][0]
                j = len(layers) - 2
                for _, layer in reversed_enumerate(layers[1:-1]):
                    for k in range(layers[j].units):
                        c += 1
                        pbar.update(min(c / cc * 100.0, 100))
                        for l in range(layers[j + 1].units):
                            tp = dj_da[j + 1][l] * self.activation_func.df_dz(z[j + 1][l]) * self.layers[j + 1].w[l, k]
                            dj_da[j][k] += tp
                            for m in range(layers[j].params):
                                dj_dw[j][k, m] += tp * self.activation_func.df_dz(z[j][k]) * a[j - 1][m]
                            dj_db[j][k] += tp * self.activation_func.df_dz(z[j][k])
                        for m in range(layers[j].params):
                            layers[j].w[k, m] -= alpha * dj_dw[j][k, m]
                        layers[j].b[k] -= alpha * dj_db[j][k]
                    j -= 1
                for k in range(layers[0].units):
                    c += 1
                    # pbar.update(min(c / cc * 100.0, 100))
                    for l in range(layers[1].units):
                        tp = dj_da[1][l] * self.activation_func.df_dz(z[1][l]) * self.layers[1].w[l, k]
                        dj_da[0][k] += tp
                        for m in range(layers[0].params):
                            dj_dw[0][k, m] += tp * self.activation_func.df_dz(z[0][k]) * feature[m]
                        dj_db[0][k] = tp * self.activation_func.df_dz(z[0][k])
                    for m in range(layers[0].params):
                        dj_dw[0][k, m] -= alpha * dj_dw[0][k, m]
                self.layers = deepcopy(layers)
            pbar.finish()
        return 0

    def train_2(self, x: np.ndarray, y: np.ndarray, epoch_count: int, alpha: float, lambda_: float):
        print('WARNING: Start training ...')
        try:
            for e in range(epoch_count):
                print(f'\nEpoch {e + 1}/{epoch_count}, loss {self.get_loss(x, y)}')
                pbar = progressbar.ProgressBar(
                    widgets=[progressbar.Percentage('%(percentage)3.4f%%'), ' ', progressbar.Bar('#'), ' ',
                             progressbar.Timer(), ' ', progressbar.ETA(), ' '], maxval=100)
                c = 0
                cc = x.shape[0] * len(self.layers)
                for i in range(x.shape[0]):
                    layers = deepcopy(self.layers)
                    feature, target = x[i], y[i]
                    z, a = self.predict_verbose(feature)
                    prediction = float(self.predict(feature))
                    dj_dz = [np.zeros((layer.units,)) for layer in self.layers]
                    dj_db = [np.zeros((layer.units,)) for layer in self.layers]
                    dj_dw = [np.zeros((layer.units, layer.params)) for layer in self.layers]
                    for l in range(len(self.layers) - 1, -1, -1):
                        c += 1
                        pbar.update(c / cc * 100.0)
                        if l == len(self.layers) - 1:
                            dj_dz[l] = self.loss_func.dl_dp(prediction, target) \
                                        * self.activation_func.df_dz(z[-1])
                        else:
                            dj_dz[l] = np.dot(dj_dz[l + 1], self.layers[l + 1].w) * self.activation_func.df_dz(z[l])
                        dj_db[l] = dj_dz[l]
                        layers[l].b -= alpha * dj_db[l]
                        if l == 0:
                            dj_dw[l] = np.matmul(dj_dz[l].reshape(dj_dz[l].shape[0], 1),
                                                 feature.reshape(1, feature.shape[0]))
                        else:
                            dj_dw[l] = np.matmul(dj_dz[l].reshape(dj_dz[l].shape[0], 1),
                                                 a[l - 1].reshape(1, a[l-1].shape[0]))
                        layers[l].w -= alpha * (dj_dw[l] + lambda_ * self.layers[l].w / x.shape[0])  # L2 regularization
                    self.layers = layers
                pbar.finish()
        except KeyboardInterrupt:
            print('\nTraining process interrupted. Data since last the last epoch will be lost.\n')
