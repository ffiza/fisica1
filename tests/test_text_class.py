import unittest
import pygame
from nbody.ui.text import Text


class TestTextClass(unittest.TestCase):
    """
    Some tests for the `Text` class defined in `nbody/ui/text.py`.
    """

    def test_failed_anchor(self):
        pygame.init()
        font = pygame.font.Font(None, 12)
        try:
            Text(loc=(100, 100), font=font, value="Example",
                 color="gray50", anchor="midbottom")
        except ValueError as error:
            self.assertTrue(isinstance(error, ValueError))


if __name__ == '__main__':
    unittest.main()
