from abc import ABC, abstractmethod

class Dictable(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass 
    
    @abstractmethod
    def from_dict(self):
        pass 