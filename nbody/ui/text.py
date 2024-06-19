import pygame


class Text:
    ANCHORS = ["bottomleft", "midbottom", "bottomright",
               "topleft", "midtop", "topright"]

    def __init__(self, loc: tuple, font: pygame.font.Font,
                 value: str, color: str, anchor: str = "bottomleft") -> None:
        """
        A class to manage a text box.

        Parameters
        ----------
        loc : tuple
            The location of the text rectangle anchor point of the rectangle
            defined in `anchor`.
        font : pygame.font.Font
            The font of the text.
        value : str
            The value of the text.
        color : str
            The color of the text.
        anchor : str, optional
            The anchor of the text. Can be `bottomleft`, `midbottom`,
            `bottomright`, `topleft`, `midtop` or `topright`.
            By default `bottomleft`.
        """
        self.__font: pygame.font.Font = font
        self.__color: str = color
        self.__surf: pygame.Surface = None
        self.__value: str = None

        self.loc: tuple = loc
        self.anchor: str = anchor
        self.value: str = value

    @property
    def loc(self) -> tuple:
        return self.__loc

    @loc.setter
    def loc(self, loc: tuple) -> None:
        self.__loc = loc

    @property
    def anchor(self) -> str:
        return self.__anchor

    @anchor.setter
    def anchor(self, anchor: str) -> None:
        if anchor in Text.ANCHORS:
            self.__anchor = anchor
        else:
            raise ValueError("Invalid anchor point. Possible values are "
                             f"{Text.ANCHORS}.")

    @property
    def value(self) -> str:
        return self.__value

    @value.setter
    def value(self, value: str) -> None:
        if self.__value != value:
            self.__value = value
            self.__surf = self.__font.render(self.__value, True, self.__color)

    def draw(self, screen: pygame.Surface) -> None:
        if self.anchor == "bottomleft":
            screen.blit(self.__surf,
                        self.__surf.get_rect(bottomleft=self.loc))
        if self.anchor == "midbottom":
            screen.blit(self.__surf,
                        self.__surf.get_rect(midbottom=self.loc))
        if self.anchor == "bottomright":
            screen.blit(self.__surf,
                        self.__surf.get_rect(bottomright=self.loc))
        if self.anchor == "topleft":
            screen.blit(self.__surf,
                        self.__surf.get_rect(topleft=self.loc))
        if self.anchor == "midtop":
            screen.blit(self.__surf,
                        self.__surf.get_rect(midtop=self.loc))
        if self.anchor == "topright":
            screen.blit(self.__surf,
                        self.__surf.get_rect(topright=self.loc))
