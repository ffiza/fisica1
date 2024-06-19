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
        self.__value: str = None

        self.set_loc(loc)
        self.set_anchor(anchor)
        self.set_value(value)

    def get_loc(self) -> tuple:
        return self.__loc

    def set_loc(self, loc: tuple) -> None:
        self.__loc = loc

    def get_anchor(self) -> str:
        return self.__anchor

    def set_anchor(self, anchor: str) -> None:
        if anchor in Text.ANCHORS:
            self.__anchor = anchor
        else:
            raise ValueError("Invalid anchor point. Possible values are "
                             f"{Text.ANCHORS}.")

    def get_value(self) -> str:
        return self.__value

    def set_value(self, value: str) -> None:
        if self.__value != value:
            self.__value = value
            self.__surf = self.__font.render(self.__value, True, self.__color)

    def draw(self, screen: pygame.Surface) -> None:
        if self.get_anchor() == "bottomleft":
            screen.blit(self.__surf,
                        self.__surf.get_rect(bottomleft=self.get_loc()))
        if self.get_anchor() == "midbottom":
            screen.blit(self.__surf,
                        self.__surf.get_rect(midbottom=self.get_loc()))
        if self.get_anchor() == "bottomright":
            screen.blit(self.__surf,
                        self.__surf.get_rect(bottomright=self.get_loc()))
        if self.get_anchor() == "topleft":
            screen.blit(self.__surf,
                        self.__surf.get_rect(topleft=self.get_loc()))
        if self.get_anchor() == "midtop":
            screen.blit(self.__surf,
                        self.__surf.get_rect(midtop=self.get_loc()))
        if self.get_anchor() == "topright":
            screen.blit(self.__surf,
                        self.__surf.get_rect(topright=self.get_loc()))
