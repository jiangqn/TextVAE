import math

class KLAnnealer(object):

    def __init__(self, latent_size: int, beta: float, anneal_step: int):
        super(KLAnnealer, self).__init__()
        self.latent_size = latent_size
        self.beta = beta
        self.anneal_step = anneal_step

    def linear_anneal(self, step: int) -> float:
        return min(step, self.anneal_step) / self.anneal_step * self.beta