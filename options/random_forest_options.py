from options.options import Options


class RandomForestOptions(Options):
    def __init__(self):
        super().__init__()

        # data options
        self.train_size = 0.75

        # model options
        self.balancedRFC = True  # bool

        # hyperparameters
        self.criterion = "entropy"
        self.n_estimators = 500
        self.max_depth = None
        self.max_features = "auto"
        self.random_state = None
        self.min_samples_split = None
        self.min_samples_leaf = None
        self.class_weight = 'balanced'

