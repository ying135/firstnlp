import torch
import dict
class Beam(object):
    def __init__(self, size, n_best=1):
        self.size =size
