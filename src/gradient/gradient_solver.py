class GradientSolver:
    def __init__(self, parent, step, precision, max_iterations):
        self.parent = parent
        self.step = step
        self.precision = precision
        self.max_iterations = max_iterations

    def get_parameter(self, param):
        return self.parent.get_parameter(f"gradient_{param}")

    def solve(self):
        ...
