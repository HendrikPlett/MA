import pickle

class Dictable:

    def to_dict(self) -> dict:
        dictionary = {}
        for key, value in self.__dict__.items():
            # Value is Iterable -> Call to_dict method on each Dictable instance
            if isinstance(value, (list, tuple, set)):
                dictionary[key] = [v.to_dict() if isinstance(v, Dictable) else v for v in value]
            # Value is Dictionary -> Call to_dict method on each Dictable instance
            elif isinstance(value, dict): 
                dictionary[key] = {k: v.to_dict() if isinstance(v, Dictable) else v for k, v in value.items()}
            # Value is Dictable instance -> call to_dict method on it
            elif isinstance(value, Dictable):
                dictionary[key] = value.to_dict()
            else:
                dictionary[key] = value
        return dictionary
    
    @classmethod
    def from_dict(cls, dict_data: dict):
        instance = cls.__new__(cls) 
        for key, value in dict_data.items():
            # Value is Iterable -> Call from_dict on all Dictable instances to transform them back
            if isinstance(value, (list, tuple, set)):
                setattr(instance, key, [cls.from_dict(v) if isinstance(v, dict) else v for v in value])
            # Value is Dictionary -> Call from_dict on all Dictable instances to transform them back
            elif isinstance(value, dict):
                setattr(instance, key, {k: cls.from_dict(v) if isinstance(v, dict) else v for k, v in value.items()})
            # Value is Dictable instance -> Call from_dict method on it
            elif isinstance(value, Dictable):
                setattr(instance, key, cls.from_dict(value))
            else:
                setattr(instance, key, value)
        return instance
    
    def to_pickle(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def from_pickle(cls, file_path: str):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
