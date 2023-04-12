class Path:
    def __init__(self):
        self._root_dir = ""

    def db_root_dir(self, dataset):
        """
        Returns the root directory of a specified dataset.
        """
        if dataset == "solis":
            return self._root_dir + "/data/l2a"
        if dataset == "cityscapes":
            return self._root_dir + "/data/cityscapes"

    def __call__(self):
        """
        Returns an instance of the Path class.
        """
        return self
