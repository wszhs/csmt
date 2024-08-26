

def load_imdb():
    from keras.datasets import imdb
    from keras.preprocessing import sequence
    from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict
    max_features = 20000
    maxlen = 80  # cut texts after this number of words (among top max_features most common words)

    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)
    X_val,y_val=X_test,y_test
    mask=get_true_mask([column for column in X_train])
    
    return X_train,y_train,X_val,y_val,X_test,y_test,mask