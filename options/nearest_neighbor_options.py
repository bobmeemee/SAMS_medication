from options.options import Options


class NearestNeighborOptions(Options):
    def __init__(self):
        super().__init__()

        self.train_size = 0.5

        self.n_neighbors = 5
        self.radius = 1
        self.algorithm = 'auto'
        self.leaf_size = 30
        self.metric = 'minkowski'
        self.p = 2
        self.metric_params = None
        self.n_jobs = -1

