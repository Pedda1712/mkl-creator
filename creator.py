"""Utility for creating synthetic datasets as multiple kernels.
Created to be able to quickly generate datasets for multiple
kernel learning algorithms.
"""
from sklearn import preprocessing
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn import preprocessing
import numpy as np
import h5py


def normalize(K: np.ndarray) -> np.ndarray:
    """Normalize a kernel."""
    diag = np.diag(K)[:, np.newaxis]
    denom = np.sqrt(diag @ diag.T)
    return np.divide(K, denom)


class Kernel:
    """Kernel base class (linear kernel)."""

    def __init__(self):
        return

    def calculate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Linear kernel as base."""
        return normalize(np.dot(X1, X2.T))


class RBFKernel(Kernel):
    """Radial Basis Kernel.

    Parameters
    ----------
    gamma : kernel distance parameter
    """

    pass

    def __init__(self, gamma: float):
        self.gamma = gamma

    def calculate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Get rbf kernel."""
        sqdist = (
            np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        )
        return normalize(np.exp(-self.gamma * sqdist))


class TanhKernel(Kernel):
    """Tanh Kernel generator.

    Parameters
    ----------
    a
    b
    """

    pass

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def calculate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Get tanh kernel."""
        return normalize(np.tanh(self.a * (X1 @ X2.T) + self.b))


class PolyKernel(Kernel):
    """Polynomial kernel generator.

    $\alpha (q + <x1, x2>)^p$
    where $\alpha \is (\frac{a^2}{2})^p$
    and   $q \is (\frac{a^2}{2}) - 1

    Parameters
    ----------
    a
    p
    """

    pass

    def __init__(self, a: float, p: float):
        self.a = a
        self.p = p

    def calculate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Get poly kernel."""
        alpha = pow(pow(self.a, 2) / 2, self.p)
        q = pow(self.a, 2) - 1
        return normalize(np.pow((np.dot(X1, X2.T) + q), self.p) * alpha)


def gamma_estimate(X):
    n_features = X.shape[1]
    return 1 / (n_features * np.var(X))


def getKernels(X: np.ndarray) -> list[np.ndarray]:
    """Let the user specify kernels."""
    result: list(np.ndarray) = []
    print("available kernel types:")
    print("[l] linear kernel")
    print("[g] rbf / gaussian kernel")
    print("[t] tanh kernel")
    print("[p] polynomial kernel")
    ks = str(input('kernels?(e.g. "lgggtp")>'))
    for char in ks:
        match char:
            case "g":
                est = gamma_estimate(X)
                gamma = float(input(f'gamma for gaussian? (heuristic: {est:.3f})>'))
                result.append(RBFKernel(gamma).calculate(X, X))
                continue
            case "t":
                a = float(input("a for tanh?>"))
                b = float(input("b for tanh?>"))
                result.append(TanhKernel(a, b).calculate(X, X))
                continue
            case "p":
                a = float(input("a for polynomial?>"))
                p = float(input("p for polynomial?>"))
                result.append(PolyKernel(a, p).calculate(X, X))
            case "l" | _:
                result.append(Kernel().calculate(X, X))
                continue
    return result


def blobs():
    """Blob data generator."""
    classes = int(input("(blobs) how many classes?>"))
    features = int(input("(blobs) how many features?>"))
    samples = int(input("(blobs) how many samples?>"))
    state = int(input("(blobs) random state (int)?>"))
    X, y = make_blobs(
        n_samples=samples, n_features=features, centers=classes, random_state=state
    )
    # standardization of data
    m = X.max(axis=0)
    m[np.where(m == 0)[0]] = 1
    X = ((X - (X.max(axis=0) - X.min(axis=0)) / 2) / (m)) * 2
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return getKernels(X), y


def uci():
    """uci data getter."""
    id = int(input("(uci) id>"))
    s = fetch_ucirepo(id=id)
    X = np.array(s.data.features)
    y = np.array(s.data.targets)
    cleaning = True
    while cleaning:
        print("(uci) (cleaning) the following labels appear in targets: ")
        print(np.unique(y))
        print("(uci) (cleaning) actions: ")
        print("[1] nothing")
        print("[2] merge labels")
        print("[3] drop rows with label")
        choice = input("(default 1)>")
        try:
            choice = int(choice)
        except ValueError:
            choice = 1
        match choice:
            case 2:
                l1 = input("(uci) (cleaning) (merging) label to merge into>")
                l2 = input("(uci) (cleaning) (merging) label to merge>")
                inds1 = np.array(np.where(y == l2))[0]
                inds2 = np.array(np.where(y == l1))[0]
                y[inds1] = y[inds2][0]
            case 3:
                l1 = input("(uci) (cleaning) (drop label) label to drop>")
                inds = np.array(np.where(y != l1))[0]
                X = X[inds, :]
                y = np.ndarray.flatten(y[inds])
            case 1 | _:
                cleaning = False
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    _, y = np.unique(np.array(y), return_inverse=True)
    y = np.ndarray.flatten(y)
    return getKernels(X), y


def main():
    """Synthetic dataset generator with h5 file output."""
    print("what type of dataset do you want to create?")
    print("[1] blob data")
    print("[2] uci data")
    choice = input("(default 1)>")
    try:
        choice = int(choice)
    except ValueError:
        choice = 1
    fname = str(input("output file?>"))
    kernels = []
    targets = []
    match choice:
        case 2:
            kernels, targets = uci()
        case 1 | _:
            kernels, targets = blobs()

    hfTrain = h5py.File(fname, "w")
    names = []
    i = 1
    for K in kernels:
        name: str = "K" + str(i)
        names.append(name)
        hfTrain.create_dataset(name, data=K)
        i = i + 1
    hfTrain.create_dataset("Labels", data=[targets])
    hfTrain.create_dataset("Kernels", data=[names])


if __name__ == "__main__":
    main()
