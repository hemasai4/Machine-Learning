# EXPERIMENT 3 – ID3 DECISION TREE ALGORITHM

## Aim

To demonstrate the working of the decision tree based ID3 algorithm, using an appropriate data set for building the decision tree and applying this knowledge to classify a new sample.

## Procedure

1. Read the training data set (PlayTennis).
2. Calculate the entropy of the entire data set.
3. Calculate the information gain of every attribute and select the attribute with the highest gain as the root node.
4. Split the data set on the values of the selected attribute and repeat steps 2 and 3 for each subset with the remaining attributes, until all examples in a subset belong to one class or no attributes are left.
5. Display the decision tree and classify a new sample using it.

## Program

```python
import math
from collections import Counter

ATTRIBUTES = ["Outlook", "Temperature", "Humidity", "Wind"]
TARGET = "PlayTennis"

# PlayTennis data set (Mitchell, Machine Learning)
DATA = [
    ["Sunny",    "Hot",  "High",   "Weak",   "No"],
    ["Sunny",    "Hot",  "High",   "Strong", "No"],
    ["Overcast", "Hot",  "High",   "Weak",   "Yes"],
    ["Rain",     "Mild", "High",   "Weak",   "Yes"],
    ["Rain",     "Cool", "Normal", "Weak",   "Yes"],
    ["Rain",     "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny",    "Mild", "High",   "Weak",   "No"],
    ["Sunny",    "Cool", "Normal", "Weak",   "Yes"],
    ["Rain",     "Mild", "Normal", "Weak",   "Yes"],
    ["Sunny",    "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High",   "Strong", "Yes"],
    ["Overcast", "Hot",  "Normal", "Weak",   "Yes"],
    ["Rain",     "Mild", "High",   "Strong", "No"],
]


def entropy(rows):
    """Entropy of the class labels (last column) of the given rows."""
    total = len(rows)
    counts = Counter(r[-1] for r in rows)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def information_gain(rows, attr_index):
    """Gain(S, A) = Entropy(S) - sum(|Sv|/|S| * Entropy(Sv))."""
    total = len(rows)
    remainder = 0.0
    for value in set(r[attr_index] for r in rows):
        subset = [r for r in rows if r[attr_index] == value]
        remainder += (len(subset) / total) * entropy(subset)
    return entropy(rows) - remainder


def id3(rows, attr_indices, verbose=False, depth=0):
    labels = [r[-1] for r in rows]

    # Case 1: all examples have the same class -> leaf node
    if len(set(labels)) == 1:
        return labels[0]
    # Case 2: no attributes left -> leaf with the majority class
    if not attr_indices:
        return Counter(labels).most_common(1)[0][0]

    # Choose the attribute with the highest information gain
    gains = {i: information_gain(rows, i) for i in attr_indices}
    best = max(gains, key=gains.get)

    if verbose:
        pad = "  " * depth
        print(f"{pad}Entropy(S) = {entropy(rows):.4f}")
        for i in attr_indices:
            print(f"{pad}  Gain({ATTRIBUTES[i]}) = {gains[i]:.4f}")
        print(f"{pad}  -> best attribute: {ATTRIBUTES[best]}\n")

    tree = {ATTRIBUTES[best]: {}}
    remaining = [i for i in attr_indices if i != best]
    for value in sorted(set(r[best] for r in rows)):
        subset = [r for r in rows if r[best] == value]
        tree[ATTRIBUTES[best]][value] = id3(subset, remaining, verbose, depth + 1)
    return tree


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "-> " + tree)
        return
    attr = next(iter(tree))
    for value, subtree in tree[attr].items():
        if isinstance(subtree, dict):
            print(f"{indent}{attr} = {value}:")
            print_tree(subtree, indent + "    ")
        else:
            print(f"{indent}{attr} = {value}: {subtree}")


def classify(tree, sample):
    """Walk the tree using a {attribute: value} dictionary."""
    while isinstance(tree, dict):
        attr = next(iter(tree))
        tree = tree[attr][sample[attr]]
    return tree


if __name__ == "__main__":
    print("Training examples:", len(DATA), "\n")
    tree = id3(DATA, list(range(len(ATTRIBUTES))), verbose=True)

    print("=" * 60)
    print("Decision tree built by ID3:")
    print_tree(tree)

    new_sample = {"Outlook": "Sunny", "Temperature": "Cool",
                  "Humidity": "High", "Wind": "Strong"}
    print("\nNew sample:", new_sample)
    print("Predicted class (PlayTennis):", classify(tree, new_sample))
```

## Output

```
Training examples: 14 

Entropy(S) = 0.9403
  Gain(Outlook) = 0.2467
  Gain(Temperature) = 0.0292
  Gain(Humidity) = 0.1518
  Gain(Wind) = 0.0481
  -> best attribute: Outlook

  Entropy(S) = 0.9710
    Gain(Temperature) = 0.0200
    Gain(Humidity) = 0.0200
    Gain(Wind) = 0.9710
    -> best attribute: Wind

  Entropy(S) = 0.9710
    Gain(Temperature) = 0.5710
    Gain(Humidity) = 0.9710
    Gain(Wind) = 0.0200
    -> best attribute: Humidity

============================================================
Decision tree built by ID3:
Outlook = Overcast: Yes
Outlook = Rain:
    Wind = Strong: No
    Wind = Weak: Yes
Outlook = Sunny:
    Humidity = High: No
    Humidity = Normal: Yes

New sample: {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
Predicted class (PlayTennis): No
```

## Result

Thus, the ID3 algorithm was implemented successfully. The decision tree was built with **Outlook** as the root node, and the new sample <Sunny, Cool, High, Strong> was classified as **PlayTennis = No**.

---

# EXPERIMENT 4 – ARTIFICIAL NEURAL NETWORK USING BACKPROPAGATION

## Aim

To build an Artificial Neural Network by implementing the Backpropagation algorithm and test the same using appropriate data sets.

## Procedure

1. Initialize the weights of the input-to-hidden and hidden-to-output layers with small random values.
2. Feed the training inputs forward through the hidden layer and the output layer using the sigmoid activation function.
3. Calculate the error at the output layer and propagate it backwards to find the error at the hidden layer.
4. Update all the weights and biases using gradient descent.
5. Repeat steps 2 to 4 for a number of epochs, then test the trained network on the XOR and Iris data sets and display the accuracy.

## Program

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """Derivative of the sigmoid expressed in terms of its output a = sigmoid(z)."""
    return a * (1.0 - a)


class NeuralNetwork:
    """Feed-forward network: input -> hidden (sigmoid) -> output (sigmoid),
    trained with the Backpropagation algorithm (gradient descent on squared error)."""

    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.5, seed=1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-1, 1, (n_inputs, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.uniform(-1, 1, (n_hidden, n_outputs))
        self.b2 = np.zeros((1, n_outputs))
        self.lr = learning_rate

    def forward(self, X):
        self.hidden = sigmoid(X @ self.W1 + self.b1)
        self.output = sigmoid(self.hidden @ self.W2 + self.b2)
        return self.output

    def backward(self, X, y):
        n = X.shape[0]
        # Error term of the output units
        delta_out = (y - self.output) * sigmoid_derivative(self.output)
        # Error term of the hidden units (error propagated backwards)
        delta_hidden = (delta_out @ self.W2.T) * sigmoid_derivative(self.hidden)
        # Weight updates
        self.W2 += self.lr * self.hidden.T @ delta_out / n
        self.b2 += self.lr * delta_out.sum(axis=0, keepdims=True) / n
        self.W1 += self.lr * X.T @ delta_hidden / n
        self.b1 += self.lr * delta_hidden.sum(axis=0, keepdims=True) / n

    def train(self, X, y, epochs, print_every=None):
        for epoch in range(1, epochs + 1):
            self.forward(X)
            self.backward(X, y)
            if print_every and epoch % print_every == 0:
                loss = np.mean((y - self.output) ** 2)
                print(f"  Epoch {epoch:5d}   MSE = {loss:.6f}")

    def predict(self, X):
        return self.forward(X)


def demo_xor():
    print("=" * 60)
    print("Data set 1: XOR problem")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    nn = NeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1, learning_rate=2.0, seed=1)
    nn.train(X, y, epochs=10000, print_every=2000)

    pred = nn.predict(X)
    print("\n  Input    Target   Output    Predicted class")
    for xi, ti, pi in zip(X, y, pred):
        print(f"  {xi.astype(int)}   {int(ti[0])}      {pi[0]:.4f}    {int(pi[0] > 0.5)}")
    acc = np.mean((pred > 0.5).astype(int) == y)
    print(f"\n  Accuracy on XOR: {acc * 100:.1f}%")


def demo_iris():
    print("\n" + "=" * 60)
    print("Data set 2: Iris (150 samples, 4 features, 3 classes)")
    iris = load_iris()
    X, labels = iris.data, iris.target
    y = np.eye(3)[labels]  # one-hot encoding of the 3 classes

    X_train, X_test, y_train, y_test, l_train, l_test = train_test_split(
        X, y, labels, test_size=0.3, random_state=42, stratify=labels)

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    nn = NeuralNetwork(n_inputs=4, n_hidden=8, n_outputs=3, learning_rate=1.0, seed=1)
    nn.train(X_train, y_train, epochs=5000, print_every=1000)

    train_acc = np.mean(np.argmax(nn.predict(X_train), axis=1) == l_train)
    test_pred = np.argmax(nn.predict(X_test), axis=1)
    test_acc = np.mean(test_pred == l_test)
    print(f"\n  Training samples: {len(X_train)}   Test samples: {len(X_test)}")
    print(f"  Training accuracy: {train_acc * 100:.2f}%")
    print(f"  Test accuracy    : {test_acc * 100:.2f}%")

    print("\n  Sample test predictions:")
    for i in range(5):
        print(f"    actual = {iris.target_names[l_test[i]]:<11} "
              f"predicted = {iris.target_names[test_pred[i]]}")


if __name__ == "__main__":
    demo_xor()
    demo_iris()
```

## Output

```
============================================================
Data set 1: XOR problem
  Epoch  2000   MSE = 0.002303
  Epoch  4000   MSE = 0.000760
  Epoch  6000   MSE = 0.000446
  Epoch  8000   MSE = 0.000313
  Epoch 10000   MSE = 0.000240

  Input    Target   Output    Predicted class
  [0 0]   0      0.0106    0
  [0 1]   1      0.9850    1
  [1 0]   1      0.9838    1
  [1 1]   0      0.0190    0

  Accuracy on XOR: 100.0%

============================================================
Data set 2: Iris (150 samples, 4 features, 3 classes)
  Epoch  1000   MSE = 0.012922
  Epoch  2000   MSE = 0.007462
  Epoch  3000   MSE = 0.005568
  Epoch  4000   MSE = 0.004505
  Epoch  5000   MSE = 0.003785

  Training samples: 105   Test samples: 45
  Training accuracy: 100.00%
  Test accuracy    : 91.11%

  Sample test predictions:
    actual = virginica   predicted = virginica
    actual = versicolor  predicted = versicolor
    actual = virginica   predicted = versicolor
    actual = versicolor  predicted = versicolor
    actual = virginica   predicted = virginica
```

## Result

Thus, the Artificial Neural Network was built using the Backpropagation algorithm and tested successfully. The network learned the XOR problem with **100%** accuracy and classified the Iris test samples with **91.11%** accuracy.
