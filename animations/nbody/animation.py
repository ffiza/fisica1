import pygame
import sys
import yaml
import pandas as pd
import argparse
import numpy as np
import utils


class Animation:
    def __init__(self, config: dict) -> None:
        """
        The constructor for the Animation class.

        Parameters
        ----------
        config : dict
            A dictionary with the configuration parameters.
        """
        pygame.init()
        self.running = True

        self.config = config

        # Screen size
        self.width = self.config["width"]
        self.height = self.config["height"]

        # Setup window
        self.screen = pygame.display.set_mode(
            size=(self.width, self.height),
            flags=pygame.FULLSCREEN,
            depth=32)
        pygame.display.set_caption(self.config["window_name"])

        # Start clock
        self.clock = pygame.time.Clock()

        # Define number of particles
        self.n_bodies = len(config["masses"])

        # Read data and transform coordinates
        self.data = pd.read_csv(
            f"animations/nbody/data/{config['filename']}.csv")
        for i in range(self.n_bodies):
            self.data[f"xPosition{i}"], self.data[f"yPosition{i}"] = \
                self._transform_coordinates(
                    x=self.data[f"xPosition{i}"].to_numpy(),
                    y=self.data[f"yPosition{i}"].to_numpy())

        # Define scaling factor for energy bars
        energies = self.data[["Potential", "KineticEnergy", "Energy"]]
        max_energy = np.max(np.abs(energies.to_numpy()))
        self.factor = self.height / max_energy / 4

        # Setup fonts
        self.font = pygame.font.SysFont("arial", 30)

        # Directory for frames for movie
        self.frames_dir = "animations/nbody/movies/frames/"

    @staticmethod
    def _quit() -> None:
        """
        This method quits the animation and closes the window.
        """
        pygame.quit()
        sys.exit()

    def _check_events(self) -> None:
        """
        Handle the events loop.
        """
        for event in pygame.event.get():
            # Quitting
            if event.type == pygame.QUIT:
                self._quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._quit()

            # Pause/Unpause
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.running = not self.running

    def _transform_coordinates(self,
                               x: np.ndarray,
                               y: np.ndarray,
                               ) -> tuple:
        """
        This method transforms simulation coordinates to PyGame coordinates.

        Parameters
        ----------
        x : np.ndarray
            The coordinates in the simulation x-axis.
        y : np.ndarray
            The coordinates in the simulation y-axis.

        Returns
        -------

        x_pyg : np.ndarray
            The coordinates in the animation x-axis.
        y_pyg : np.ndarray
            The coordinates in the animation y-axis.
        """
        movie_xrange = np.diff(self.config["scene_xlim"])
        movie_yrange = np.diff(self.config["scene_ylim"])
        x_pyg = self.width / movie_xrange * x + self.width / 2
        y_pyg = - self.height / movie_yrange * y + self.height / 2
        return x_pyg, y_pyg

    def _draw_indicator_lines(self) -> None:
        """
        Draw lines to guide the eye.
        """
        pygame.draw.aaline(
            surface=self.screen,
            color=self.config["ind_color"],
            start_pos=(0, self.height / 2),
            end_pos=(self.width, self.height / 2))
        pygame.draw.aaline(
            surface=self.screen,
            color=self.config["ind_color"],
            start_pos=(self.width / 2, 0),
            end_pos=(self.width / 2, self.height))

    def _draw_energy_bars(self, idx: int) -> None:
        """
        Draw the energy bars of the system.

        Parameters
        ----------
        idx : int
            The index of the data frame to plot.
        """
        # Define geometrical quantities
        bar_width = self.config["bar_width_frac"] * self.width
        bar_sep = self.config["bar_sep_frac"] * self.width

        # Draw energy bars
        x0 = self.config["text_start_frac"] * self.width
        for energy in ["Energy", "Potential", "KineticEnergy"]:
            y0 = self.height / 2 \
                - abs(self.data[energy].iloc[idx]) * self.factor
            bar_height = abs(self.data[energy].iloc[idx]) * self.factor
            if self.data[energy].iloc[idx] < 0:
                y0 += abs(self.data[energy].iloc[idx]) * self.factor
            pygame.draw.rect(
                self.screen,
                self.config["ind_color"],
                pygame.Rect(
                    x0,
                    y0,
                    bar_width,
                    bar_height + 1))
            # The +1 in the previous line fixes minor visualization issues
            x0 += bar_sep

        # Draw energy labels
        x0 = self.config["text_start_frac"] * self.width
        dy = self.config["text_offset"] * self.height
        letters = ["E", "U", "K"]
        for i, energy in enumerate(["Energy", "Potential", "KineticEnergy"]):
            text = self.font.render(letters[i],
                                    True,
                                    self.config["ind_color"])
            if self.data[energy].iloc[idx] >= 0:
                self.screen.blit(
                    text,
                    text.get_rect(
                        midtop=(x0 + bar_width / 2, self.height / 2 + dy)))
            else:
                self.screen.blit(
                    text,
                    text.get_rect(
                        midbottom=(x0 + bar_width / 2, self.height / 2 - dy)))
            x0 += bar_sep

    def _draw_particles(self, idx: int) -> None:
        """
        Draw the particles.

        Parameters
        ----------
        idx : int
            The index of the data frame to plot.
        """
        for i in range(self.n_bodies):
            if self.n_bodies <= len(self.config["part_colors"]):
                color = self.config["part_colors"][i]
            else:
                color = self.config["part_colors"][0]
            if idx >= 1 \
                    and self.n_bodies <= len(self.config["part_colors"]):
                # Trace of the particle
                pygame.draw.aalines(
                    surface=self.screen,
                    color=color,
                    closed=False,
                    points=np.vstack(
                        (self.data[f"xPosition{i}"].iloc[:idx],
                         self.data[f"yPosition{i}"].iloc[:idx])).T,
                )
            # Particle as sphere
            pygame.draw.circle(
                self.screen,
                color,
                (self.data[f"xPosition{i}"].iloc[idx],
                 self.data[f"yPosition{i}"].iloc[idx]),
                7)

    def _draw_energy_and_time_values(self, idx: int) -> None:
        """
        Draw the energy and time values.

        Parameters
        ----------
        idx : int
            The index of the data frame to plot.
        """
        x0 = self.config["text_start_frac"] * self.width
        text = self.font.render(
            f"Energy: {self.data['Energy'].iloc[idx]:.2f} J",
            True,
            self.config["ind_color"])
        self.screen.blit(
            text,
            text.get_rect(bottomleft=(x0, self.height - 2 * x0)))
        text = self.font.render(
            f"Time: {self.data['Time'].iloc[idx]:.1f} s",
            True,
            self.config["ind_color"])
        self.screen.blit(
            text,
            text.get_rect(bottomleft=(x0, self.height - x0)))

    def _draw_elements(self, idx: int) -> None:
        """
        Draw the elements on the screen.

        Parameters
        ----------
        idx : int
            The index of the data frame to plot.
        """

        self._draw_indicator_lines()
        self._draw_energy_bars(idx=idx)
        self._draw_energy_and_time_values(idx=idx)
        self._draw_particles(idx=idx)

    def _create_frames_dir(self):
        """
        This method creates a directory to store the frames for the movie.
        """
        if self.config["save_frames"]:
            utils.create_dir(path=f"{self.frames_dir}")

    def _save_screen_as_img(self, frame_idx: int):
        """
        This method saves the current screen as an image with index
        `frame_idx`.

        Parameters
        ----------
        frame_idx : int
            The index of this frame.
        """
        pygame.image.save(self.screen,
                          f"{self.frames_dir}frame{frame_idx}.png")

    def run(self) -> None:
        """
        Run the main animation loop.
        """

        # Create folder for frames if necessary
        self._create_frames_dir()

        idx = 0
        while True:  # Main game loop
            self._check_events()
            if idx >= len(self.data):
                # Quit the animation if simulation reaches the end
                self._quit()
            self.screen.fill(self.config["bg_color"])
            self._draw_elements(idx=idx)
            self.clock.tick(self.config["fps"])

            if self.running:
                # Create frames if `save_frames` is True
                if self.config["save_frames"]:
                    self._save_screen_as_img(
                        frame_idx=int(idx // self.config["one_every"]))

                # Advance to the next frame
                idx += self.config["one_every"]
            else:
                text = self.font.render(
                    "Paused",
                    True,
                    self.config["ind_color"])
                self.screen.blit(
                    text,
                    text.get_rect(topright=(self.width - 0.05 * self.height,
                                            0.05 * self.height)))
            pygame.display.flip()


def main():
    # Get simulation name
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation",
                        type=str,
                        required=True,
                        help="The simulation to animate.")
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(
        open(f"animations/nbody/configs/{args.simulation}.yml"))

    # Run the PyGame animation
    animation = Animation(config=config)
    animation.run()


if __name__ == "__main__":
    main()
