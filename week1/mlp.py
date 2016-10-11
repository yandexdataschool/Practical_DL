import theano.tensor as T

from logistic_regression import LogisticRegression
from hidden_layer import HiddenLayer


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        #
        #         self.L1 = (
        #             abs(self.hiddenLayer.W).sum()
        #             + abs(self.logRegressionLayer.W).sum()
        #         )

        #         self.L2_sqr = (
        #             (self.hiddenLayer.W ** 2).sum()
        #             + (self.logRegressionLayer.W ** 2).sum()
        #         )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input