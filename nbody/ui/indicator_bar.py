import pygame


class IndicatorBar:
    def __init__(self, left: int, top: int, width: int, height: int,
                 color: str, fill_direction: str) -> None:
        self.__left: int = left
        self.__top: int = top
        self.__width: int = width
        self.__height: int = height
        self.__color: str = color

        self.__rect: pygame.Rect = pygame.Rect(
            self.__left, self.__top, self.__width, self.__height)

        self.fill_direction: str = fill_direction
        self.value: float = 1.0

    @property
    def fill_direction(self) -> str:
        return self.__fill_direction

    @fill_direction.setter
    def fill_direction(self, new_fill_direction: str) -> None:
        if new_fill_direction in ["vertical", "horizontal"]:
            self.__fill_direction = new_fill_direction
        else:
            raise ValueError("Invalid `fill_direction`. Must be either "
                             "`vertical` or `horizontal`.")

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    def value(self, new_value: float) -> None:

        if new_value > 1.0:
            new_value = 1.0
        if new_value < -1.0:
            new_value = -1.0
        self.__value = new_value

        if self.__fill_direction == "horizontal":
            self.__rect.width = self.__value * self.__width
            if self.__rect.left != self.__left:
                self.__rect.left = self.__left
        if self.__fill_direction == "vertical":
            self.__rect.height = self.__value * self.__height
            if self.__rect.top != self.__top:
                self.__rect.top = self.__top

        if self.__value < 0.0:
            self.__rect.normalize()

    @property
    def rect(self) -> pygame.Rect:
        return self.__rect

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, self.__color, self.__rect)
