class Path:
    def __init__(self):
        self._root_dir = ""

    def db_root_dir(dataset):
        """
        Returns the root directory of a specified dataset.
        """
        if dataset == "solis":
            # TODO: Change this to your own path
            return "/opt/data/copernicus"
        if dataset == "cityscapes":
            return "/home/erling/enernite/autodeeplab/data/cityscapes"

    def __call__(self):
        """
        Returns an instance of the Path class.
        """
        return self
