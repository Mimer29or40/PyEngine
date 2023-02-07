from abc import ABC
from abc import abstractmethod


class AbstractEngine(ABC):
    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def update(self, time: float, delta_time: float) -> None:
        pass

    @abstractmethod
    def draw(self, time: float, delta_time: float) -> None:
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass
