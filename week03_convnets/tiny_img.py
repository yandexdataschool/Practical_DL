import numpy as np
from sklearn.model_selection import train_test_split
import os, sys
if sys.version_info[0] == 2:
    from urllib import urlretrieve
    import cPickle as pickle

else:
    from urllib.request import urlretrieve
    import pickle

def unpickle(file):
    fo = open(file, 'rb')
    if sys.version_info[0] == 2:
        dict = pickle.load(fo)
    else:
        dict = pickle.load(fo,encoding='latin1')
    
    fo.close()
    return dict

def download_tinyImg200(path,
                     url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                     tarname='tiny-imagenet-200.zip'):
    if not os.path.exists(path):
        os.mkdir(path)
    urlretrieve(url, os.path.join(path,tarname))
    print (os.path.join(path,tarname))
    import zipfile
    zip_ref = zipfile.ZipFile(os.path.join(path,tarname), 'r')
    zip_ref.extractall()
    zip_ref.close()
    
    
from PIL import Image

def read_folder(folder_path):
    list_of_pics = [Image.open(os.path.join(folder_path, filename)).getdata() for filename in os.listdir(folder_path) if np.array(Image.open(os.path.join(folder_path, filename)).getdata()).shape == (4096, 3)]
    return np.array(list_of_pics).reshape(np.array(list_of_pics).shape[0], 64, 64, 3)

def load_tiny_image(data_path=".", channels_last=False, test_size=0.3, random_state=1337):
    data_path = '.'
    full_data_path = os.path.join(data_path, "tiny-imagenet-200/")
    list_of_folders = open(os.path.join(full_data_path, "wnids.txt"), 'r')
    folder_names = [line.split() for line in list_of_folders.readlines()]
    data_paths = [os.path.join(data_path, "tiny-imagenet-200/train/" + elem[0] + "/images") for elem in folder_names]

    if not os.path.exists(full_data_path) or not all(list(map(os.path.exists, data_paths))):
        print ("Dataset not found. Downloading...")
        print (data_path)
        download_tinyImg200(data_path)

    X_list = [read_folder(path) for path in data_paths]  
    X = np.concatenate(X_list).reshape([-1,64,64,3]).astype('float32')/255
    y_list = []

    for class_label in np.arange(len(data_paths)):
        print (len(X_list[class_label]))
        y_list.append(np.full((len(X_list[class_label]),), class_label))

    y = np.concatenate(y_list).astype('int32')
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                   test_size=test_size,
                                                   random_state=random_state)
    
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                   test_size=test_size,
                                                   random_state=random_state)
    
    print ("shapes: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape )

    return X_train,y_train,X_val,y_val,X_test,y_test


def look_at_class(data, labels):
    class_n = random.randint(1,200)
    idxs = []
    print ("class ", class_n)
    for ind, label in enumerate(labels):
        if label == class_n:
            idxs.append(ind)
            
    data_for_show = data[idxs]
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(data_for_show[i].reshape(64,64,3))
    

def look_up_same_classes(data, labels, number_of_classes = 4):
    for i in range(number_of_classes):
        look_at_class(data, labels)
    return 0