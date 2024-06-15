import pygame
import yaml

CONFIG = yaml.safe_load(open("configs/global.yml"))


class Debugger:
    def __init__(self):
        self.font = pygame.font.SysFont("arial", 15)
        self._reset_msg()
        self.text_sep = CONFIG["DEBUG_TEXT_SEP"]

    def _reset_msg(self):
        self.msgs = [""]

    def render(self, msgs: list, screen: pygame.Surface):
        """
        Blit the debug surface to the screen

        Parameters
        ----------
        screen : pygame.Surface
            The screen onto which to blit the surface.
        """
        self._reset_msg()
        for i, msg in enumerate(msgs):
            self.surf = self.font.render(msg, True, 'white')
            self.rect = self.surf.get_rect(topleft=(0, i * self.text_sep))
            screen.blit(self.surf, self.rect)
