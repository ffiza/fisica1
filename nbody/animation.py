import pygame
import sys
import yaml
import pandas as pd
import argparse
import numpy as np

import utils
from ui.debugger import Debugger
from ui.indicator_bar import IndicatorBar
from ui.grid import Grid

PARTICLE_COLORS = [
    "#FA4656", "#2C73D6", "#00D75B", "#FEF058", "#FFAA4C", "#A241B2"]


class Animation:
    def __init__(self, df: pd.DataFrame) -> None:
        """
        The constructor for the Animation class.

        Parameters
        ----------
        df : pd.DataFrame
            The data to animate.
        """
        pygame.init()
        self.running = False
        self.debugging = True
        self.debugger = Debugger()
        self.config = yaml.safe_load(open("configs/global.yml"))
        self.data = df
        self.n_bodies = utils.get_particle_count_from_df(self.data)
        self.n_frames = len(self.data)
        self.initial_energy = self.data['Energy'].iloc[0]
        self.idx = 0  # Current simulation snapshot

        # Setup window
        self.screen = pygame.display.set_mode(
            size=(self.config["SCREEN_WIDTH"], self.config["SCREEN_HEIGHT"]),
            flags=pygame.FULLSCREEN, depth=32)
        pygame.display.set_caption(self.config["WINDOW_NAME"])

        self.clock = pygame.time.Clock()

        # Read data and transform coordinates
        for i in range(self.n_bodies):
            self.data[f"xPosition{i}"], self.data[f"yPosition{i}"] = \
                self._transform_coordinates(
                    x=self.data[f"xPosition{i}"].to_numpy(),
                    y=self.data[f"yPosition{i}"].to_numpy())

        # Setup fonts
        font = self.config["FONT"]
        self.font = pygame.font.Font(f"fonts/{font}.ttf", 30)

        # Setup grid
        self.grid = Grid(
            sep=self.config["GRID_SEPARATION_PX"],
            xmin=0.0, xmax=self.config["SCREEN_WIDTH"],
            ymin=0.0, ymax=self.config["SCREEN_HEIGHT"],
            width=1, color=self.config["GRID_COLOR"],
            major_lw_factor=self.config["GRID_MAJOR_LW_FACTOR"])

        # Define geometry for energy bars
        energies = self.data[["Potential", "KineticEnergy", "Energy"]]
        self.min_energy = np.min(energies)
        self.max_energy = np.max(energies)
        self.max_energy_abs = np.max(np.abs(energies))
        self.factor = self.config["SCREEN_HEIGHT"] / self.max_energy_abs / 4
        bar_sep = self.config["BAR_SEP_FRAC"] * self.config["SCREEN_WIDTH"]
        bar_left = self.config["TEXT_START_FRAC"] * self.config["SCREEN_WIDTH"]
        bar_base_level = self.config["SCREEN_HEIGHT"] / 2
        bar_width = self.config["BAR_WIDTH_FRAC"] * self.config["SCREEN_WIDTH"]
        bar_height = self.config["SCREEN_HEIGHT"] / 4

        # Setup energy bars
        self.mechanical_energy_bar = IndicatorBar(
            left=bar_left, base_level=bar_base_level,
            width=bar_width, height=bar_height,
            color=self.config["INDICATORS_COLOR"], label="E", font=self.font,
            text_sep=self.config["TEXT_OFFSET"] * self.config["SCREEN_HEIGHT"])
        self.potential_energy_bar = IndicatorBar(
            left=bar_left + bar_sep, base_level=bar_base_level,
            width=bar_width, height=bar_height,
            color=self.config["INDICATORS_COLOR"], label="U", font=self.font,
            text_sep=self.config["TEXT_OFFSET"] * self.config["SCREEN_HEIGHT"])
        self.kinetic_energy_bar = IndicatorBar(
            left=bar_left + 2 * bar_sep, base_level=bar_base_level,
            width=bar_width, height=bar_height,
            color=self.config["INDICATORS_COLOR"], label="K", font=self.font,
            text_sep=self.config["TEXT_OFFSET"] * self.config["SCREEN_HEIGHT"])

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
        self.idx = 0
        self.running = False

    def _draw_indicator_lines(self) -> None:
        """
        Draw lines to guide the eye.
        """
        pygame.draw.line(
            surface=self.screen,
            color=self.config["INDICATORS_COLOR"],
            start_pos=(0, self.config["SCREEN_HEIGHT"] / 2),
            end_pos=(self.config["SCREEN_WIDTH"],
                     self.config["SCREEN_HEIGHT"] / 2))
        pygame.draw.line(
            surface=self.screen,
            color=self.config["INDICATORS_COLOR"],
            start_pos=(self.config["SCREEN_WIDTH"] / 2, 0),
            end_pos=(self.config["SCREEN_WIDTH"] / 2,
                     self.config["SCREEN_HEIGHT"]))

    def _update_energy_values(self) -> None:
        """
        Update the values of the energy bars to the current snapshot index.
        """
        self.mechanical_energy_bar.set_value(
            self.data['Energy'].iloc[self.idx] / self.max_energy_abs)
        self.potential_energy_bar.set_value(
            self.data['Potential'].iloc[self.idx] / self.max_energy_abs)
        self.kinetic_energy_bar.set_value(
            self.data['KineticEnergy'].iloc[self.idx] / self.max_energy_abs)

    def _draw_energy_bars(self) -> None:
        """
        Draw the energy bars of the system.
        """
        self.mechanical_energy_bar.draw(self.screen)
        self.potential_energy_bar.draw(self.screen)
        self.kinetic_energy_bar.draw(self.screen)

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
                    surface=self.screen,
                    color=color,
                    closed=False,
                    points=np.vstack(
                        (self.data[f"xPosition{i}"].iloc[:self.idx],
                         self.data[f"yPosition{i}"].iloc[:self.idx])).T,
                )
            # Particle as sphere
            pygame.draw.circle(
                self.screen,
                color,
                (self.data[f"xPosition{i}"].iloc[self.idx],
                 self.data[f"yPosition{i}"].iloc[self.idx]),
                10)

    def _draw_energy_and_time_values(self) -> None:
        """
        Draw the energy and time values in the current snapshot.
        """
        x0 = self.config["TEXT_START_FRAC"] * self.config["SCREEN_WIDTH"]
        text = self.font.render(
            f"Energy: {self.data['Energy'].iloc[self.idx]:.2f} J",
            True, self.config["INDICATORS_COLOR"])
        self.screen.blit(
            text,
            text.get_rect(
                bottomleft=(x0, self.config["SCREEN_HEIGHT"] - 2 * x0)))
        text = self.font.render(
            f"Time: {self.data['Time'].iloc[self.idx]:.1f} s",
            True, self.config["INDICATORS_COLOR"])
        self.screen.blit(
            text,
            text.get_rect(
                bottomleft=(x0, self.config["SCREEN_HEIGHT"] - 1 * x0)))

    def _draw_elements(self) -> None:
        """
        Draw the elements on the screen in the current snapshot.
        """

        self._draw_indicator_lines()
        self.grid.draw(self.screen)
        self._draw_energy_bars()
        self._draw_energy_and_time_values()
        self._draw_particles()

    def run(self) -> None:
        """
        Run the main animation loop.
        """

        while True:  # Main game loop
            self._check_events()
            if self.idx >= len(self.data):
                self._reset_animation()

            self._update_energy_values()

            self.screen.fill(self.config["BACKGROUND_COLOR"])
            self._draw_elements()

            self.clock.tick(self.config["FPS"])

            if self.debugging:
                self.debugger.render(
                    [f"FPS: {self.clock.get_fps()}",
                     f"DEBUGGING: {int(self.debugging)}",
                     f"N_PARTICLES: {self.n_bodies}",
                     f"CURRENT_SNAPSHOT_IDX: {self.idx}",
                     f"MAX_SNAPSHOT_IDX: {len(self.data) - 1}",
                     f"TIME: {self.data['Time'].iloc[self.idx]}",
                     f"MAX_TIME: {self.data['Time'].iloc[-1]}",
                     f"MEC_ENERGY: {self.data['Energy'].iloc[self.idx]}",
                     f"POT_ENERGY: {self.data['Potential'].iloc[self.idx]}",
                     f"KIN_ENERGY: {self.data['KineticEnergy'].iloc[self.idx]}",
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


def main():
    # Get simulation name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", type=str, required=True,
        help="The simulation to animate.")
    args = parser.parse_args()

    # Load configuration file
    df = pd.read_csv(f"results/{args.result}.csv")

    # Run the PyGame animation
    animation = Animation(df=df)
    animation.run()


if __name__ == "__main__":
    main()
