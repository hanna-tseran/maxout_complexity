class RegionFunction:
    def __init__(self, W=[], b=[]):
        self.W = W
        self.b = b

class Region:
    def __init__(self, function=None, next_layer_function=None, lhs_inequalities=[], rhs_inequalities=[],
        lhs_equalities=[], rhs_equalities=[]):
        self.function = function
        self.next_layer_function = next_layer_function
        self.lhs_inequalities = lhs_inequalities
        self.rhs_inequalities = rhs_inequalities
        self.lhs_equalities = lhs_equalities
        self.rhs_equalities = rhs_equalities
