# EXPERIMENT 1 – FIND-S ALGORITHM

## Aim

To implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples.

## Procedure

1. Initialize the hypothesis h to the most specific hypothesis.
2. Read the training examples one by one.
3. Ignore the negative examples.
4. For each positive example, compare every attribute with the hypothesis: copy the value if the hypothesis holds `0`, and replace it with `?` if the values differ.
5. Display the final most specific hypothesis.

## Program

```python
ATTRIBUTES = ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast"]

DATA = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same",   "Yes"],
    ["Sunny", "Warm", "High",   "Strong", "Warm", "Same",   "Yes"],
    ["Rainy", "Cold", "High",   "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High",   "Strong", "Cool", "Change", "Yes"],
]


def find_s(data, positive_label="Yes", verbose=True):
    n_attrs = len(data[0]) - 1
    h = ["0"] * n_attrs  # most specific hypothesis
    if verbose:
        print(f"Initial hypothesis: {h}\n")

    for i, row in enumerate(data, start=1):
        x, label = row[:-1], row[-1]

        if label != positive_label:
            if verbose:
                print(f"Example {i}: {x} -> {label} (negative, ignored)")
                print(f"  h = {h}\n")
            continue

        for j in range(n_attrs):
            if h[j] == "0":          # first positive example: copy it
                h[j] = x[j]
            elif h[j] != x[j]:       # mismatch: generalize
                h[j] = "?"

        if verbose:
            print(f"Example {i}: {x} -> {label} (positive)")
            print(f"  h = {h}\n")

    return h


if __name__ == "__main__":
    final_h = find_s(DATA)
    print("Most specific hypothesis found by FIND-S:")
    print(dict(zip(ATTRIBUTES, final_h)))
    print(final_h)
```

## Output

```
Initial hypothesis: ['0', '0', '0', '0', '0', '0']

Example 1: ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'] -> Yes (positive)
  h = ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']

Example 2: ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'] -> Yes (positive)
  h = ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']

Example 3: ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'] -> No (negative, ignored)
  h = ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']

Example 4: ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'] -> Yes (positive)
  h = ['Sunny', 'Warm', '?', 'Strong', '?', '?']

Most specific hypothesis found by FIND-S:
{'Sky': 'Sunny', 'AirTemp': 'Warm', 'Humidity': '?', 'Wind': 'Strong', 'Water': '?', 'Forecast': '?'}
['Sunny', 'Warm', '?', 'Strong', '?', '?']
```

## Result

Thus, the FIND-S algorithm was implemented successfully and the most specific hypothesis obtained was **<Sunny, Warm, ?, Strong, ?, ?>**.

---

# EXPERIMENT 2 – CANDIDATE-ELIMINATION ALGORITHM

## Aim

To implement and demonstrate the Candidate-Elimination algorithm for a given set of training data examples stored in a .CSV file, and to output a description of the set of all hypotheses consistent with the training examples.

## Procedure

1. Read the training examples from the .CSV file.
2. Initialize the specific boundary S to the most specific hypothesis and the general boundary G to the most general hypothesis.
3. For each positive example, remove the inconsistent hypotheses from G and minimally generalize the hypotheses in S.
4. For each negative example, remove the inconsistent hypotheses from S and minimally specialize the hypotheses in G.
5. Display the final S and G boundaries and all the hypotheses consistent with the training examples.

## Training Data (training_data.csv)

```
Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes
```

## Program

```python
import csv
import os
import sys
from itertools import product


def covers(h, x):
    """True if hypothesis h classifies instance x as positive."""
    return all(hv == "?" or (hv != "0" and hv == xv) for hv, xv in zip(h, x))


def more_general_or_equal(h1, h2):
    """True if h1 is more general than or equal to h2."""
    for a, b in zip(h1, h2):
        if a == "?":
            continue
        if b == "0":
            continue
        if a == "0" or a != b:
            return False
    return True


def min_generalization(s, x):
    """Minimal generalization of s that covers positive instance x."""
    return tuple(
        xv if sv == "0" else (sv if sv == xv else "?") for sv, xv in zip(s, x)
    )


def min_specializations(g, x, domains):
    """Minimal specializations of g that exclude negative instance x."""
    result = []
    for j, gv in enumerate(g):
        if gv == "?":
            for v in domains[j]:
                if v != x[j]:
                    h = list(g)
                    h[j] = v
                    result.append(tuple(h))
    return result


def load_csv(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    header, body = rows[0], rows[1:]
    attributes = [c.strip() for c in header[:-1]]
    data = [([v.strip() for v in r[:-1]], r[-1].strip()) for r in body]
    return attributes, data


def candidate_elimination(data, n_attrs, positive_label="Yes", verbose=True):
    domains = [sorted({x[j] for x, _ in data}) for j in range(n_attrs)]
    S = {tuple(["0"] * n_attrs)}
    G = {tuple(["?"] * n_attrs)}

    def show(title):
        if verbose:
            print(f"  {title}")
            print(f"    S = {sorted(S)}")
            print(f"    G = {sorted(G)}\n")

    show("Initial boundaries")

    for i, (x, label) in enumerate(data, start=1):
        x = tuple(x)
        if label == positive_label:
            if verbose:
                print(f"Example {i}: {list(x)} -> {label} (positive)")
            G = {g for g in G if covers(g, x)}
            for s in list(S):
                if not covers(s, x):
                    S.remove(s)
                    h = min_generalization(s, x)
                    if any(more_general_or_equal(g, h) for g in G):
                        S.add(h)
            S = {s for s in S
                 if not any(s != t and more_general_or_equal(s, t) for t in S)}
        else:
            if verbose:
                print(f"Example {i}: {list(x)} -> {label} (negative)")
            S = {s for s in S if not covers(s, x)}
            for g in list(G):
                if covers(g, x):
                    G.remove(g)
                    for h in min_specializations(g, x, domains):
                        if any(more_general_or_equal(h, s) for s in S):
                            G.add(h)
            G = {g for g in G
                 if not any(g != t and more_general_or_equal(t, g) for t in G)}
        show("After example")

    return S, G, domains


def enumerate_version_space(data, domains, positive_label="Yes"):
    """Brute-force every hypothesis consistent with all examples (cross-check)."""
    choices = [d + ["?"] for d in domains]
    consistent = []
    for h in product(*choices):
        if all(covers(h, x) == (label == positive_label) for x, label in data):
            consistent.append(h)
    return consistent


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "training_data.csv")
    path = sys.argv[1] if len(sys.argv) > 1 else default

    attributes, data = load_csv(path)
    print(f"Attributes: {attributes}")
    print(f"Training examples: {len(data)}\n")

    S, G, domains = candidate_elimination(data, len(attributes))

    print("=" * 60)
    print("Final specific boundary S:")
    for s in sorted(S):
        print("  ", list(s))
    print("Final general boundary G:")
    for g in sorted(G):
        print("  ", list(g))

    vs = enumerate_version_space(data, domains)
    print(f"\nAll {len(vs)} hypotheses in the version space (between S and G):")
    for h in sorted(vs):
        print("  ", list(h))
```

## Output

```
Attributes: ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast']
Training examples: 4

  Initial boundaries
    S = [('0', '0', '0', '0', '0', '0')]
    G = [('?', '?', '?', '?', '?', '?')]

Example 1: ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'] -> Yes (positive)
  After example
    S = [('Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same')]
    G = [('?', '?', '?', '?', '?', '?')]

Example 2: ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'] -> Yes (positive)
  After example
    S = [('Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same')]
    G = [('?', '?', '?', '?', '?', '?')]

Example 3: ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'] -> No (negative)
  After example
    S = [('Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same')]
    G = [('?', '?', '?', '?', '?', 'Same'), ('?', 'Warm', '?', '?', '?', '?'), ('Sunny', '?', '?', '?', '?', '?')]

Example 4: ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'] -> Yes (positive)
  After example
    S = [('Sunny', 'Warm', '?', 'Strong', '?', '?')]
    G = [('?', 'Warm', '?', '?', '?', '?'), ('Sunny', '?', '?', '?', '?', '?')]

============================================================
Final specific boundary S:
   ['Sunny', 'Warm', '?', 'Strong', '?', '?']
Final general boundary G:
   ['?', 'Warm', '?', '?', '?', '?']
   ['Sunny', '?', '?', '?', '?', '?']

All 6 hypotheses in the version space (between S and G):
   ['?', 'Warm', '?', '?', '?', '?']
   ['?', 'Warm', '?', 'Strong', '?', '?']
   ['Sunny', '?', '?', '?', '?', '?']
   ['Sunny', '?', '?', 'Strong', '?', '?']
   ['Sunny', 'Warm', '?', '?', '?', '?']
   ['Sunny', 'Warm', '?', 'Strong', '?', '?']
```

## Result

Thus, the Candidate-Elimination algorithm was implemented successfully. The specific boundary obtained was **S = {<Sunny, Warm, ?, Strong, ?, ?>}**, the general boundary was **G = {<Sunny, ?, ?, ?, ?, ?>, <?, Warm, ?, ?, ?, ?>}**, and the version space contained 6 consistent hypotheses.
