
def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    import cPickle
    data = cPickle.load(f)
    f.close()
    return data 