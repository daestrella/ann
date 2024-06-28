import matplotlib.pyplot as plt

def graph(figsize, title, legend, xlabel, ylabel, x, y=None):
    plt.figure(figsize=figsize)
    if y is not None:
        plt.plot(x, y, label=legend)
    else:
        plt.plot(x, label=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

