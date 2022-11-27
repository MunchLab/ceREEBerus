import numpy as np
def torus(num = 50):
    """Create torus object
    
        args:
            num (int): number of samples to use when creating torus data

        returns:
            3-element tuple containing

            - **X** : X-coordinates
            - **Y** : Y-coordinates
            - **Z** : Z-coordinates

    """

    U = np.linspace(0, 2*np.pi, num)
    V = np.linspace(0, 2*np.pi, num)
    U, V = np.meshgrid(U, V)
    X = (np.cos(V))*np.cos(U)
    Y = (np.cos(V))*np.sin(U)
    Z = np.sin(V)
    return X, Y, Z