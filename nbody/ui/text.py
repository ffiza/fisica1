import pygame


class Text:
    def __init__(self, loc: tuple, font: pygame.font.Font,
                 value: str, color: str) -> None:
        """
        A class to manage a text box.

        Parameters
        ----------
        loc : tuple
            The location of the bottom-left corner of the text rectangle.
        font : pygame.font.Font
            The font of the text.
        value : str
            The value of the text.
        color : str
            The color of the text.
        """
        self.loc: tuple = loc
        self.font: pygame.font.Font = font
        self.color: str = color

        self.value: str = None
        self.set_value(value)

    def set_value(self, new_value: str) -> None:
        if self.value != new_value:
            self.value = new_value
            self.surf = self.font.render(self.value, True, self.color)

    def draw(self, screen: pygame.Surface) -> None:
        screen.blit(self.surf, self.surf.get_rect(bottomleft=self.loc))
