import pygame
import sys
import numpy as np

from ui.grid import Grid
from ui.indicator_bar import NewIndicatorBar


class BlackScreen:
    SCREEN_SIZE: tuple = (800, 600)
    COLOR_DEPTH: int = 32
    WINDOW_NAME: str = "Black Screen"
    FONT_SIZE: int = 12
    FPS: int = 60
    BACKGROUND_COLOR: str = "gray10"

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            size=BlackScreen.SCREEN_SIZE,
            flags=pygame.NOFRAME,
            depth=BlackScreen.COLOR_DEPTH)
        pygame.display.set_caption(BlackScreen.WINDOW_NAME)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, BlackScreen.FONT_SIZE)

        self.bar = NewIndicatorBar(
            left=400, top=300, width=200, height=10, color="gray50",
            fill_direction="horizontal")

        self.time: float = 0.0
        self.dt: float = 0.0

        self.grid = Grid(sep=50, xmin=0, xmax=BlackScreen.SCREEN_SIZE[0],
                         ymin=0, ymax=BlackScreen.SCREEN_SIZE[1],
                         width=1, color="gray25", major_lw_factor=3)

    @staticmethod
    def _quit() -> None:
        """
        Quit the animation and close the window.
        """
        pygame.quit()
        sys.exit()

    def _handle_user_events(self) -> None:
        """
        Handle the user inputs.
        """
        for event in pygame.event.get():
            # Quitting
            if event.type == pygame.QUIT:
                BlackScreen._quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                BlackScreen._quit()

    def run(self) -> None:
        """
        Run the main animation loop.
        """

        while True:  # Main game loop
            self._handle_user_events()

            self.time += self.dt
            self.bar.value = np.sin(self.time)

            self.screen.fill(BlackScreen.BACKGROUND_COLOR)
            self.grid.draw(self.screen)
            self.bar.draw(self.screen)

            pygame.display.flip()

            self.dt = self.clock.tick(BlackScreen.FPS) / 1000


if __name__ == "__main__":
    animation = BlackScreen()
    animation.run()
