from options.options import Options


class LogisticRegressionOptions(Options):
    def __init__(self):
        super().__init__()

        # data options
        self.train_size = 0.5

        self.class_weight = None


    def set_train_size(self, train_size):
        self.train_size = train_size
