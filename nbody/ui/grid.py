import pygame


class Grid:
    """
    A class to manage an a grid.
    """
    def __init__(self, sep: float, xmin: float, xmax: float,
                 ymin: float, ymax: float, width: int, color: tuple,
                 major_lw_factor: int):
        self.width: int = width
        self.color: tuple = color
        self.major_lw_factor: int = major_lw_factor

        center: tuple = ((xmax + xmin) / 2, (ymax + ymin) / 2)

        x = center[0]
        y = center[1]
        self.nx: int = 0  # This is needed for line width
        self.start_poss: list = []
        self.end_poss: list = []
        while x < xmax:
            self.start_poss.append((x, ymin))
            self.end_poss.append((x, ymax))
            if x > center[0]:
                self.start_poss.append((2 * center[0] - x, ymin))
                self.end_poss.append((2 * center[0] - x, ymax))
            x += sep
            self.nx += 1
        while y < ymax:
            self.start_poss.append((xmin, y))
            self.end_poss.append((xmax, y))
            if y > center[1]:
                self.start_poss.append((xmin, 2 * center[1] - y))
                self.end_poss.append((xmax, 2 * center[1] - y))
            y += sep

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draws the grid on `screen`.

        Parameters
        ----------
        screen : pygame.Surface
            The screen on which to draw the grid.
        """
        for i in range(len(self.start_poss)):
            width_factor: int = 1
            if (i == 0) or (i == 2 * self.nx - 1):
                width_factor = self.major_lw_factor
            pygame.draw.line(
                surface=screen, color=self.color,
                width=width_factor * self.width,
                start_pos=self.start_poss[i], end_pos=self.end_poss[i])

    def show(self) -> None:
        """
        Print start and end points for testing.
        """
        for i in range(len(self.start_poss)):
            print(self.start_poss[i], self.end_poss[i])
