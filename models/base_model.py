from abc import ABC,abstractmethod

class Model(ABC):
    @abstractmethod
    def fit(self,*arg,**kwarg):
        pass
    @abstractmethod
    def predict(self):
        pass
    