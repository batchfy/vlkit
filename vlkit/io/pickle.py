
import pickle


def save_pkl(object, path):
    """save python object to a pickle file

    Example:
    ::

        # save a dict to file
        dic = dict(a=1, b=2)
        save_pkl(dic, "dic.pkl")
        print(load_pkl("dic.pkl"))


    :param object: the python object to be saved.
    :param path: target path.
    """
    f = open(path, "wb")
    pickle.dump(object, f)
    f.close()
    return path

def load_pkl(path):
    """load python object from a pickle file
    """
    f = open(path, "rb")
    return pickle.load(f)
