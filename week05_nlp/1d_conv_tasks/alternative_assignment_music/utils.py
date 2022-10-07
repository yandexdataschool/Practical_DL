import time
import numpy as np

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
        
def train_net(net, train_fun, test_fun, X_train, y_train, X_valid, y_valid, num_epochs=10, batch_size=5):
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train,batch_size):
            inputs, targets = batch
            train_err_batch, train_acc_batch= train_fun(inputs, targets)
            train_err += train_err_batch
            train_acc += train_acc_batch
            train_batches += 1

        # And a full pass over the validation data:
        val_acc = 0
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_valid, y_valid, batch_size):
            inputs, targets = batch
            val_err_batch, val_acc_batch = test_fun(inputs, targets)
            val_acc  += val_acc_batch
            val_err  += val_err_batch
            val_batches += 1

        # Then we print the results for this epoch:
        ou = "Epoch %3s of %3s train_loss = %.2f val_loss = %.2f train_acc = %.2f val_acc = %.2f"
        print(ou % (epoch + 1, num_epochs, 
                    train_err / train_batches, val_err / val_batches, 
                    train_acc / train_batches * 100, val_acc / val_batches * 100))
    return net