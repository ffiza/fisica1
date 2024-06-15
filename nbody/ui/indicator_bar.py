import pygame


class IndicatorBar:
    """
    A class to manage an indicator bar that can change its fill value.
    """
    def __init__(self, left: float, base_level: float,
                 width: float, height: float, color: tuple, label: str,
                 font: pygame.font.Font, text_sep: float):
        self.left: float = left
        self.base_level: float = base_level
        self.height: float = height
        self.color: tuple = color
        self.text_sep: float = text_sep
        self.text_sep_dir: int = 1
        self.text: pygame.Surface = font.render(label, True, color)
        self.text_centerx: float = self.left + width / 2
        self.rect = pygame.Rect(
            self.left, self.base_level, width, self.height)

        self.value: float = None
        self.set_value(1.0)

    def set_value(self, value: float):
        """
        Sets the value of indicator bar and updates geometric quantities.

        Parameters
        ----------
        value : float
            The new value. Must be between -1.0 and 1.0.
        """
        if value > 1.0:
            value = 1.0
        if value < -1.0:
            value = -1.0

        self.value = value
        self.rect.height = abs(self.value * self.height)
        if self.value < 0.0:
            self.rect.top = self.base_level
            self.text_sep_dir = -1
        else:
            self.rect.top = self.base_level - self.rect.height
            self.text_sep_dir = 1

    def draw(self, screen: pygame.Surface):
        """
        Draws the indicator bar and its label on `screen`.

        Parameters
        ----------
        screen : pygame.Surface
            The screen on which to draw the indicator bar.
        """
        pygame.draw.rect(screen, self.color, self.rect)
        screen.blit(
            self.text,
            self.text.get_rect(
                centerx=self.text_centerx,
                centery=self.base_level + self.text_sep_dir * self.text_sep))
