import chainer
from chainer.backends import cuda

import chainer.links as L
import chainer.functions as F

import numpy as np

from copy import copy

class ExampleModel(chainer.Chain):
    def __init__(self, c: int):
        super().__init__()
        self.constant_member = np.asarray(c)
        self._persistent.add("constant_member")
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            self.conv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=False)
            self.dcnv1 = L.DeconvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        cuda.to_cpu(self.constant_member) # if necessary

    def to_gpu(self, device=None):
        super().to_gpu()
        cuda.to_gpu(self.constant_member) # if necessary

    def calc(self, x, target):
        original_shape = list(x.shape)
        target_shape = copy(original_shape)
        target_shape.insert(1, 1)
        m = F.relu(self.conv1(F.reshape(x, target_shape)))
        o = self.dcnv1(m)
        return F.mean_squared_error(target, F.reshape(o, original_shape))

    def __call__(self, x):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss