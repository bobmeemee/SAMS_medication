from options.options import Options


class DecisiontreeOptions(Options):
    def __init__(self):
        super().__init__()

        # data options
        self.train_size = 0.4

        # hyperparameters
        self.criterion = "entropy"
        self.max_depth = 4
        self.splitter = "best"
        self.max_features = "auto"
        self.min_samples_split = None
        self.min_samples_leaf = None
        self.ccp_alpha = 0  # minimal cost comlexity pruned
        self.class_weight = 'balanced'

    def set_train_size(self, train_size):
        self.train_size = train_size
