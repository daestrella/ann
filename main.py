from ann import Network
import numpy as np

network = Network(layers=(2, 1), lrate=0.01, init_method='He Normal', activation='ReLU', debug=True)
network.train(in_file='input.csv', out_file='output.csv', max_epoch=10000, min_error=1e-15)

while True:
    if input('Do you want to feed inputs to the network? [Y/n]: ') == 'n':
        break

    try:
        print(network.predict(np.array([
            float(input("Enter first number: ")),
            float(input("Enter second number: "))
            ])))
    except ValueError:
        print("Invalid input. Please enter a valid number...")
    except Exception as e:
        print(f'Error: {e}')
        
