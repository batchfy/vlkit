import yaml
from os.path import dirname, abspath, join
def imagenet_labels():
    data_path = abspath(join(dirname(__file__), "data/imagenet1000_clsidx_to_labels.txt"))

    with open(data_path, "r") as f:
        s = f.read()

    imagenet_labels = yaml.load(s)
    return imagenet_labels
