import random
import skimage
import skimage.transform
import numpy as np
from lasagne.utils import floatX

MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

# Returns a list of tuples (cnn features, list of words, image ID)
def get_data_batch(dataset, size, sl, split='train'):
    items = []
    
    while len(items) < size:
        item = random.choice(dataset)
        if item['split'] != split:
            continue
        sentence = random.choice(item['sentences'])['tokens']
        if len(sentence) > sl:
            continue
        items.append((item['cnn features'], sentence, item['cocoid']))
    
    return items

# Convert a list of tuples into arrays that can be fed into the network
def prep_batch_for_network(batch, sl, word_to_index):
    x_cnn = floatX(np.zeros((len(batch), 1000)))
    x_sentence = np.zeros((len(batch), sl - 1), dtype='int32')
    y_sentence = np.zeros((len(batch), sl), dtype='int32')
    mask = np.zeros((len(batch), sl), dtype='bool')

    for j, (cnn_features, sentence, _) in enumerate(batch):
        x_cnn[j] = cnn_features
        i = 0
        for word in ['#START#'] + sentence + ['#END#']:
            if word in word_to_index:
                mask[j, i] = True
                y_sentence[j, i] = word_to_index[word]
                x_sentence[j, i] = word_to_index[word]
                i += 1
                
    return x_cnn, x_sentence, y_sentence, mask

def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (224, w*224/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*224/w, 224), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])