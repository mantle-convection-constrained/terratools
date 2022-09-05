import numpy as np

class TerraModel():
    """
    Class holding a TERRA model at a single point in time.
    """

    def __init__(self, nlayers):
        self.nlayers = nlayers
        self.radii = np.linspace(3480, 6370, self.nlayers)

