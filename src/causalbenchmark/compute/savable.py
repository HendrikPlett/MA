import pickle
import os

class Pickable:
    """A base class providing pickle functionality to each subclass."""

    def __init__(self, name: str):
        """
        Initialize a Pickable object.

        Args:
            name (str): The name of the object, used as the pickle file name.
        """
        self._name = name

    def pickle(self, pre_path: str = "results"):
        """
        Save the object via pickle.

        Args:
            pre_path (str): The directory to save the pickle file in. Defaults to "results".
        """
        name = os.path.join(pre_path, f"{self._name}.pkl")
        os.makedirs(pre_path, exist_ok=True)
        if os.path.exists(name):
            print(f"Info: Existing file '{name}' will be overwritten.")
        with open(name, 'wb') as file:
            pickle.dump(self, file)
        print(f"{self.__class__.__name__} pickled under {name}")


    @classmethod
    def unpickle(cls, path: str):
        """
        Load a pickled object from a file.

        Args:
            path (str): The path to the pickle file.

        Returns:
            The unpickled object.

        Raises:
            ValueError: If the specified path does not exist.
        """
        if not os.path.exists(path):
            raise ValueError("Path does not exist.")
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        print(f"{obj.__class__.__name__} unpickled from {path}")
        return obj



class Dictable:
    # TODO: Test if this works
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
