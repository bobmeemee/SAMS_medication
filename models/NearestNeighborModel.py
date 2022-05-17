from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from models.Model import Model
from options.nearest_neighbor_options import NearestNeighborOptions


class NearestNeighborModel(Model):
    def __init__(self, data, options: NearestNeighborOptions):
        super().__init__(data)
        self.options = options

        X = data[options.feature_col]  # Features
        y = data[options.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=options.train_size,
                                                                                random_state=options.random_state,
                                                                                stratify=y)

        self.clf = NearestNeighbors(n_neighbors=self.options.n_neighbors, radius=self.options.radius,
                                    leaf_size=options.leaf_size, metric=options.metric, p=options.p,
                                    metric_params= options.metric_params, n_jobs=options.n_jobs)

    def scale_model(self, scaler):
        scaler.fit(self.X_train)
        Standardized_X_train = scaler.transform(self.X_train)
        Standardized_X_test = scaler.transform(self.X_test)

        # necessary?
        self.X_train = Standardized_X_train
        self.X_test = Standardized_X_test

    def train_model(self):
        self.clf = self.clf.fit(self.X_train, self.y_train)

    def test_model(self, isMultilabel: bool):
        A = self.clf.kneighbors_graph(self.X_train)
        A.toarray()
        print(A)
        pass




