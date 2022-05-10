from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def scale_model(self, scaler):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def test_model(self, isMultilabel: bool):
        pass



