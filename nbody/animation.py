import pygame
import sys
import yaml
import pandas as pd
import argparse
import numpy as np

import utils
from ui.debugger import Debugger
from ui.indicator_bar import IndicatorBar
from ui.indicator_bar import HorizontalIndicatorBar
from ui.grid import Grid
from ui.text import Text

PARTICLE_COLORS = [
    "#FA4656", "#2C73D6", "#00D75B", "#FEF058", "#FFAA4C", "#A241B2"]


class Animation:
    def __init__(self, xpos: np.ndarray, ypos: np.ndarray,
                 time: np.ndarray, energies: np.ndarray) -> None:
        """
        The constructor for the Animation class.

        Parameters
        ----------
        xpos : np.ndarray
            The position of the particles in the x-axis. `pos[i, j]` is the
            position of particle `j` at snapshot `i`.
        ypos : np.ndarray
            The position of the particles in the y-axis. `pos[i, j]` is the
            position of particle `j` at snapshot `i`.
        time : np.ndarray
            An array with the time of each snapshot.
        energies : np.ndarray
            A 2D array with the energy at each snapshot. `energies[i, 0]` is
            the mechanical energy at snapshot `i`; `energies[i, 1]` is the
            total kinetic energy at snapshot `i`; `energies[i, 2]` is the
            total potential energy at snapshot `i`.
        """
        self.xpos: np.ndarray = xpos
        self.ypos: np.ndarray = ypos
        self.time: np.ndarray = time
        self.energies: np.ndarray = energies

        self.max_time: float = self.time[-1]

        pygame.init()

        self.config: dict = yaml.safe_load(open("configs/global.yml"))
        self.running = int(self.config["ANIMATION_STARTUP_RUN_STATE"])
        self.debugging = int(self.config["ANIMATION_STARTUP_DEBUG_STATE"])
        self.debugger = Debugger()
        self.n_frames = self.xpos.shape[0]
        self.n_bodies = self.xpos.shape[1]
        self.initial_energy = self.energies[0, 0]
        self.idx = 0  # Current simulation snapshot

        # Transform coordinates
        for i in range(self.n_bodies):
            self.xpos[:, i], self.ypos[:, i] = \
                self._transform_coordinates(
                    x=self.xpos[:, i], y=self.ypos[:, i])

        # Setup window
        self.screen = pygame.display.set_mode(
            size=(self.config["SCREEN_WIDTH"], self.config["SCREEN_HEIGHT"]),
            flags=pygame.FULLSCREEN, depth=self.config["COLOR_DEPTH"])
        pygame.display.set_caption(self.config["WINDOW_NAME"])

        # Setup clock
        self.clock = pygame.time.Clock()

        # Setup fonts
        font = self.config["FONT"]
        self.font = pygame.font.Font(f"fonts/{font}.ttf",
                                     self.config["FONT_SIZE"])

        # Setup grid
        self.grid = Grid(
            sep=self.config["GRID_SEPARATION_PX"],
            xmin=0.0, xmax=self.config["SCREEN_WIDTH"],
            ymin=0.0, ymax=self.config["SCREEN_HEIGHT"],
            width=self.config["GRID_MINOR_LW_PX"],
            color=self.config["GRID_COLOR"],
            major_lw_factor=self.config["GRID_MAJOR_LW_FACTOR"])

        # Define geometry for energy bars
        self.min_energy = np.min(self.energies)
        self.max_energy = np.max(self.energies)
        self.max_energy_abs = np.max(np.abs(self.energies))
        self.factor = self.config["SCREEN_HEIGHT"] / self.max_energy_abs / 4
        bar_sep = self.config["BAR_SEP_FRAC"] * self.config["SCREEN_WIDTH"]
        self.ind_x0 = self.config["TEXT_START_FRAC"] \
            * self.config["SCREEN_WIDTH"]
        bar_base_level = self.config["SCREEN_HEIGHT"] / 2
        bar_width = self.config["BAR_WIDTH_FRAC"] * self.config["SCREEN_WIDTH"]
        bar_height = self.config["SCREEN_HEIGHT"] / 4

        # Setup energy bars
        self.mechanical_energy_bar = IndicatorBar(
            left=self.ind_x0, base_level=bar_base_level,
            width=bar_width, height=bar_height,
            color=self.config["INDICATORS_COLOR"], label="E", font=self.font,
            text_sep=self.config["TEXT_OFFSET"] * self.config["SCREEN_HEIGHT"])
        self.potential_energy_bar = IndicatorBar(
            left=self.ind_x0 + bar_sep, base_level=bar_base_level,
            width=bar_width, height=bar_height,
            color=self.config["INDICATORS_COLOR"], label="U", font=self.font,
            text_sep=self.config["TEXT_OFFSET"] * self.config["SCREEN_HEIGHT"])
        self.kinetic_energy_bar = IndicatorBar(
            left=self.ind_x0 + 2 * bar_sep, base_level=bar_base_level,
            width=bar_width, height=bar_height,
            color=self.config["INDICATORS_COLOR"], label="K", font=self.font,
            text_sep=self.config["TEXT_OFFSET"] * self.config["SCREEN_HEIGHT"])

        # Setup time bar
        self.time_bar = HorizontalIndicatorBar(
            left=0,
            top=self.config["SCREEN_HEIGHT"] - self.config["TIME_BAR_HEIGHT"],
            width=self.config["SCREEN_WIDTH"],
            height=self.config["TIME_BAR_HEIGHT"],
            color=self.config["INDICATORS_COLOR"], font=self.font)

        # Set energy and time text boxes
        self.energy_text = Text(
            loc=(self.ind_x0, self.config["SCREEN_HEIGHT"] - 2 * self.ind_x0),
            font=self.font, value=f"{self.energies[0, 0]:.2f}",
            color=self.config["INDICATORS_COLOR"])
        self.time_text = Text(
            loc=(self.ind_x0, self.config["SCREEN_HEIGHT"] - 1 * self.ind_x0),
            font=self.font, value=f"{self.time[0]:.2f}",
            color=self.config["INDICATORS_COLOR"])

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
                self._quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._quit()

            # Pause/Unpause
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.running = not self.running

            # Reset simulation
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._reset_animation()

            # Enable debugging
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.debugging = not self.debugging

            # Frame control by user
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                if self.idx <= self.n_frames - 2:
                    self.idx += self.config["FRAME_SKIP"]
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                if self.idx >= 1:
                    self.idx -= self.config["FRAME_SKIP"]

    def _transform_coordinates(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Transform simulation coordinates to PyGame coordinates.

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
        movie_xrange = np.diff(
            [self.config["SCENE_XMIN"], self.config["SCENE_XMAX"]])
        movie_yrange = np.diff(
            [self.config["SCENE_YMIN"], self.config["SCENE_YMAX"]])
        x_pyg = self.config["SCREEN_WIDTH"] / movie_xrange * x \
            + self.config["SCREEN_WIDTH"] / 2
        y_pyg = - self.config["SCREEN_HEIGHT"] / movie_yrange * y \
            + self.config["SCREEN_HEIGHT"] / 2
        return x_pyg, y_pyg

    def _reset_animation(self) -> None:
        """
        Reset the animation to the first snapshot and pause the run.
        """
        self.idx = 0
        self.running = False

    def _update_indicator_bars(self) -> None:
        """
        Update the values of the energy and time bars to the current snapshot
        index.
        """
        self.mechanical_energy_bar.set_value(
            self.energies[self.idx, 0] / self.max_energy_abs)
        self.potential_energy_bar.set_value(
            self.energies[self.idx, 2] / self.max_energy_abs)
        self.kinetic_energy_bar.set_value(
            self.energies[self.idx, 1] / self.max_energy_abs)
        self.time_bar.set_value(self.time[self.idx] / self.max_time)

    def _update_text(self) -> None:
        """
        Update the values of the energy and text elements to the current
        snapshot.
        """
        self.energy_text.set_value(
            f"Energy: {self.energies[self.idx, 0]:.2f} J")
        self.time_text.set_value(f"Time: {self.time[self.idx]:.1f} s")

    def _draw_bars(self) -> None:
        """
        Draw the energy and time bars.
        """
        self.mechanical_energy_bar.draw(self.screen)
        self.potential_energy_bar.draw(self.screen)
        self.kinetic_energy_bar.draw(self.screen)
        self.time_bar.draw(self.screen)

    def _draw_particles(self) -> None:
        """
        Draw the particles in the current snapshot.
        """
        for i in range(self.n_bodies):
            if self.n_bodies <= len(PARTICLE_COLORS):
                color = PARTICLE_COLORS[i]
            else:
                color = PARTICLE_COLORS[0]
            if self.idx >= 5 \
                    and self.n_bodies <= len(PARTICLE_COLORS):
                # Trace of the particle
                pygame.draw.aalines(
                    surface=self.screen, color=color, closed=False,
                    points=np.vstack((self.xpos[:self.idx, i],
                                      self.ypos[:self.idx, i])).T)
            # Particle as sphere
            pygame.draw.circle(self.screen, color,
                               (self.xpos[self.idx, i],
                                self.ypos[self.idx, i]), 10)

    def _draw_text(self) -> None:
        """
        Draw the energy and time values in the current snapshot.
        """
        self.energy_text.draw(self.screen)
        self.time_text.draw(self.screen)

    def _draw_elements(self) -> None:
        """
        Draw the elements on the screen in the current snapshot.
        """

        self.grid.draw(self.screen)
        self._draw_bars()
        self._draw_text()
        self._draw_particles()

    def run(self) -> None:
        """
        Run the main animation loop.
        """

        while True:  # Main game loop
            self._handle_user_events()
            if self.idx >= self.n_frames:
                self._reset_animation()

            self._update_indicator_bars()
            self._update_text()

            self.screen.fill(self.config["BACKGROUND_COLOR"])
            self._draw_elements()

            if self.debugging:
                self.debugger.render(
                    [f"FPS: {self.clock.get_fps()}",
                     f"DEBUGGING: {int(self.debugging)}",
                     f"N_PARTICLES: {self.n_bodies}",
                     f"CURRENT_SNAPSHOT_IDX: {self.idx}",
                     f"MAX_SNAPSHOT_IDX: {self.n_frames - 1}",
                     f"TIME: {self.time[self.idx]}",
                     f"MAX_TIME: {self.time[-1]}",
                     f"MEC_ENERGY: {self.energies[self.idx, 0]}",
                     f"POT_ENERGY: {self.energies[self.idx, 2]}",
                     f"KIN_ENERGY: {self.energies[self.idx, 1]}",
                     f"MIN_ENERGY: {self.min_energy}",
                     f"MAX_ENERGY: {self.max_energy}",
                     f"MAX_ENERGY_ABS: {self.max_energy_abs}",
                     ],
                    self.screen)

            if self.running:
                self.idx += 1
            else:
                text = self.font.render(
                    "Paused", True, self.config["INDICATORS_COLOR"])
                self.screen.blit(
                    text,
                    text.get_rect(
                        topright=(self.config["SCREEN_WIDTH"]
                                  - 0.05 * self.config["SCREEN_HEIGHT"],
                                  0.05 * self.config["SCREEN_HEIGHT"])))

            pygame.display.flip()

            self.clock.tick(self.config["FPS"])


def main():
    # Get simulation name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", type=str, required=True,
        help="The simulation to animate.")
    args = parser.parse_args()

    # Load configuration file
    df = pd.read_csv(f"results/{args.result}.csv")
    n_bodies = utils.get_particle_count_from_df(df)
    xpos = df[[f"xPosition{i}" for i in range(n_bodies)]].to_numpy()
    ypos = df[[f"yPosition{i}" for i in range(n_bodies)]].to_numpy()
    time = df["Time"].to_numpy()
    energies = df[["Energy", "KineticEnergy", "Potential"]].to_numpy()

    # Run the PyGame animation
    animation = Animation(xpos=xpos, ypos=ypos, time=time, energies=energies)
    animation.run()


if __name__ == "__main__":
    main()
