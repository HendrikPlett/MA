from abc import ABC, abstractmethod

class Dictable(ABC):

    def __init__(self):
        pass

    def to_dict(self):
        return self.__dict__
    
    @abstractmethod
    def from_dict(self):
        pass 