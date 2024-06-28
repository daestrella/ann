import numpy as np

functions = {
        "ReLU": {
            "function": lambda x: np.maximum(0, x),
            "derivative": lambda x: np.where(x > 0, 1, 0)
            },
        "identity": {
            "function": lambda x: x,
            "derivative": lambda x: np.ones_like(x)
            },
        "binary": {
            "function": lambda x: np.where(x < 0, 0, 1),
            "derivative": lambda x: np.zeros_like(x)
            },
        "softplus": {
            "function": lambda x: np.log(1 + np.exp(x)),
            "derivative": lambda x: 1 / (1 + np.exp(x))
            },
        "tansig": {
            "function": lambda x: np.tanh(x),
            "derivative": lambda x: 1 - (np.tanh(x)) ** 2
            }
        }

loss = {
        "MSE": {
            "function": lambda values: np.mean([(target-actual)**2 for target, actual in values]),
            "derivative": lambda values: (2 / len(values)) * np.sum([(actual - target) for target, actual in values])
            }
        }

# i := inputs; o := outputs
init = {
        "Zero": lambda i, o: 0,
        "Xavier": lambda i, o: np.random.uniform(-np.sqrt(6)/(i+o), np.sqrt(6)/(i+o)),
        "Xavier Normal": lambda i, o: np.random.normal(0, np.sqrt(2 / (i + o))),
        "He": lambda i, o: np.random.uniform(-np.sqrt(6)/i, np.sqrt(6)/i),
        "He Normal": lambda i, o: np.random.normal(0, np.sqrt(2 / i))
        }
