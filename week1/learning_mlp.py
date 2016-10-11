import numpy
import timeit

import theano
import theano.tensor as T

from mnist import load_dataset
from learning_logistic_regression import shared_dataset
from mlp import MLP


def test_mlp(learning_rate=0.01,
             # L1_reg=0.00, L2_reg=0.0001,
             n_epochs=300,
             dataset=load_dataset(), batch_size=20, n_hidden=500):
    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = dataset

    train_set_x = numpy.reshape(train_set_x, (train_set_x.shape[0], 28 * 28))
    valid_set_x = numpy.reshape(valid_set_x, (valid_set_x.shape[0], 28 * 28))
    test_set_x = numpy.reshape(test_set_x, (test_set_x.shape[0], 28 * 28))

    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    cost = (
        classifier.negative_log_likelihood(y)
        #         + L1_reg * classifier.L1
        #         + L2_reg * classifier.L2_sqr
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
        ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training')

    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
