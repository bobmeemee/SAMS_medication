from options.options import Options


class RandomForestOptions(Options):
    def __init__(self):
        super().__init__()

        # data options
        self.train_size = 0.5

        # model options
        self.balancedRFC = True  # bool

        # hyperparameters
        self.criterion = "gini"
        self.n_estimators = 100
        self.max_depth = 6
        self.max_features = "auto"
        self.random_state = None
        self.min_samples_split = None
        self.min_samples_leaf = None
