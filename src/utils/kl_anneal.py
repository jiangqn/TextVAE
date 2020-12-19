import math

class KLAnnealer(object):

    def __init__(self, beta: float, anneal_type: str, anneal_step: int, offset: float = 5):
        super(KLAnnealer, self).__init__()
        assert anneal_type in ["linear", "sigmoid"]
        self.beta = beta
        self.anneal_type = anneal_type
        self.anneal_step = anneal_step
        self.offset = offset
        self.sigmoid = lambda x: 1 / (1 + math.exp(-x))

    def _linear_anneal(self, step: int) -> float:
        return min(step, self.anneal_step) / self.anneal_step

    def _sigmoid_anneal(self, step: int) -> float:
        return self.sigmoid(step * 2 * self.offset / self.anneal_step - self.offset)

    def anneal(self, step: int) -> float:
        if self.anneal_type == "linear":
            return self._linear_anneal(step) * self.beta
        else:   # self.anneal_type == "sigmoid"
            return self._sigmoid_anneal(step) * self.beta