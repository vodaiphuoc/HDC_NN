import numpy as np
from typing import Union, List, Literal
from abc import ABC, abstractmethod
from copy import deepcopy

class Base_HPV(ABC):
    """
    Base class for implementing operations of HPVs
    """
    def __init__(self, 
                 D:int = 10000, 
                 counter_dtype = np.int16,
                 hpv_dtype = np.int8,
                 input_sequence: Union[List, np.ndarray] = None
                 ) -> None:
        """
        Args:
            - D(int): dimension of a hpv
            - counter_dtype : minimum is np.int16
            - hpv_dtype : np.int8
        """
        if input_sequence is None:
            self.D = D
            self.hpv_dtype = hpv_dtype
            self.counter_dtype = counter_dtype
            # self._counter = np.random.choice(a= [1,-1],size = D, replace= True).astype(dtype)
            self._counter = np.zeros(shape= D, dtype= counter_dtype)
            self._hpv = np.empty(shape= D, dtype= hpv_dtype)
        else:
            if isinstance(input_sequence, List):
                assert isinstance(input_sequence[0], int), f"Expect type int, found {type(input_sequence[0])}"
                self.D = len(input_sequence)
            elif isinstance(input_sequence, np.ndarray):
                assert input_sequence.dtype == np.int8, f"Expect type np.int8, found {input_sequence.dtype}"
                self.D = input_sequence.shape[0]
            else:
                raise NotImplementedError
            self.hpv_dtype = hpv_dtype
            self.counter_dtype = counter_dtype
            self._counter = deepcopy(input_sequence).astype(counter_dtype)
            self._hpv = deepcopy(input_sequence).astype(hpv_dtype)

    @abstractmethod
    def add(self) -> None:
        pass

    @abstractmethod
    def accumulate(self) -> None:
        pass

    @abstractmethod
    def binding(self) -> None:
        pass

    def get(self) -> np.ndarray:
        return self._hpv

    def __repr__(self) -> str:
        return f"{self.get()}"

    def shape(self) -> int:
        return self.get().shape[0]

    def count_element(self, value:Literal[0,1,-1]):
        return np.where(self.get() == value)[0].shape[0]

class Bipolar_HPV(Base_HPV):
    """
    Implement of bipolar hpv (contain only 1 and -1)
    """
    def __init__(self, 
                 D:int = 10000, 
                 counter_dtype = np.int16,
                 hpv_dtype = np.int8,
                 input_sequence: Union[List, np.ndarray] = None,
                 name:str = None
                 ) -> None:
        """
        Args:
            - D(int): dimension of a hpv
            - dtype : np.int8
        """
        super().__init__(D= D, 
                         counter_dtype = counter_dtype,
                         hpv_dtype = hpv_dtype,
                         input_sequence = input_sequence)
        self.name = name if name is not None else None

    def add(self, input_hpv: Union[np.ndarray, Base_HPV]) -> None:
        """
        Add a hpv into _counter variable
        """
        if isinstance(input_hpv, Base_HPV):
            np.add(self._counter, input_hpv.get().astype(self.counter_dtype), out= self._counter)
        else:
            np.add(self._counter, input_hpv.astype(self.counter_dtype), out= self._counter)
        return None

    def accumulate(self) -> None:
        """
        A method for accumulating bits from _counter variable
        Need to call this function after end of many add other arrays
        """
        count_random = 0
        for dim in range(self.D):
            if self._counter[dim] > 0:
                self._hpv[dim] = 1
            elif self._counter[dim] < 0:
                self._hpv[dim] = -1
            else:
                self._hpv[dim] = np.random.choice(a = [1,-1], size = None, p = [0.5,0.5])
                count_random += 1
        if self.name is not None:
            print(f"HPV name: {self.name}, number random dim: {count_random}")
        return None

    def binding(self, 
                input_hpv: Union[np.ndarray, Base_HPV]
                ) -> Union[np.ndarray, Base_HPV]:
        """
        Binding operation for bipolar vectors
        return 1 if two elements are equal, else -1
        """
        if isinstance(input_hpv, Base_HPV):
            return Bipolar_HPV(input_sequence = np.multiply(self.get(), input_hpv.get()).astype(self.hpv_dtype))
        else:
            return Bipolar_HPV(input_sequence = np.multiply(self.get(), input_hpv).astype(self.hpv_dtype))

    def flip(self) -> None:
        """
        Flip bits in all dimension of hpv
        ex: 1 is converted to -1
        """
        np.multiply(self.get(), -1, out= self._hpv)

    def flip_by_mask(self, mask: Base_HPV) -> None:
        """
        Flip hpv with mask which contains 1 or -1, where
        1 for no update, -1 for updating
        """
        np.multiply(self.get(), mask.get(), out= self._hpv)


class Binary_HPV(Base_HPV):
    """
    Implement of bipolar hpv (contain only 1 and 0)
    """
    def __init__(self, 
                 D:int = 10000, 
                 counter_dtype = np.int16,
                 hpv_dtype = np.uint8,
                 input_sequence: Union[List, np.ndarray] = None,
                 name:str = None
                 ) -> None:
        """
        Args:
            - D(int): dimension of a hpv
            - dtype : np.uint8
        """
        super().__init__(D= D, 
                         counter_dtype = counter_dtype,
                         hpv_dtype = hpv_dtype,
                         input_sequence = input_sequence)
        self.name = name if name is not None else None