import unittest
import pygame
from nbody.ui.text import Text


class TestTextClass(unittest.TestCase):
    """
    Some tests for the `Text` class defined in `nbody/ui/text.py`.
    """

    def test_failed_anchor_1(self):
        pygame.init()
        font = pygame.font.Font(None, 12)
        try:
            Text(loc=(100, 100), font=font, value="Example",
                 color="gray50", anchor="midbottom")
        except ValueError as error:
            self.assertTrue(isinstance(error, ValueError))

    def test_failed_anchor_2(self):
        pygame.init()
        font = pygame.font.Font(None, 12)
        try:
            Text(loc=(100, 100), font=font, value="Example",
                 color="gray50", anchor="middlebottom")
        except ValueError as error:
            self.assertTrue(isinstance(error, ValueError))

    def test_failed_anchor_with_setter(self):
        pygame.init()
        font = pygame.font.Font(None, 12)
        text = Text(loc=(100, 100), font=font, value="Example",
                    color="gray50", anchor="midbottom")
        try:
            text.set_anchor("middlebottom")
        except ValueError as error:
            self.assertTrue(isinstance(error, ValueError))


if __name__ == '__main__':
    unittest.main()
