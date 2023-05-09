import vpython as vp
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import time

from colors import Colors

# Device properties
MASS: float = 1.0  # kg
LENGTH: float = 1.5  # m
GRAVITY: float = 9.8  # m/s^2
THETA0: float = np.radians(0.0)  # radians
# OMEGA0: float = 0.0  # 1/s
OMEGA0: float = np.sqrt(2 * GRAVITY / LENGTH)  # 1/s

# Scene properties
DEVICE_HEIGHT: float = 1.0  # m
TIME_STEP = 0.005  # s
FINAL_TIME: float = 60.0  # s
FRAMERATE = int(1 / TIME_STEP)
SCENE_WIDTH: int = 600  # px
SCENE_HEIGHT: int = 600  # px
SCENE_RANGE: float = 2.0  # m
CYLINDRICAL_FRAME_SCALE: float = 0.5
DYNAMICS_SCALE: float = 0.15
FORCES_SCALE: float = 0.05

# Grid configuration
XMIN: float = -0.4
XMAX: float = 2.4
YMIN: float = -1.6
YMAX: float = 1.6
STEP: float = 0.2


class Vector(vp.vector):
    # TODO: Add documentation.
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y, z)
        self.transform()

    def transform(self):
        # TODO: Add documentation.
        x = self.x
        y = self.y
        z = self.z

        self.x = y
        self.y = DEVICE_HEIGHT - x
        self.z = z


class Displacement(vp.vector):
    # TODO: Add documentation.
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y, z)
        self.transform()

    def transform(self):
        # TODO: Add documentation.
        x = self.x
        y = self.y
        z = self.z

        self.x = y
        self.y = -x
        self.z = z


class Solver:
    # TODO: Add documentation.
    def __init__(self):
        pass

    def _equations(self, initial_conds: list, times: np.ndarray):
        # TODO: Add documentation.
        theta, omega = initial_conds
        f = [omega, - GRAVITY / LENGTH * np.sin(theta)]
        return f

    def solve(self, tf: float, dt: float, theta0: float, omega0: float):
        # TODO: Add documentation.
        initial_conds = [theta0, omega0]
        times = np.arange(0, tf, dt)

        # Solve the differential equations
        solution = odeint(self._equations, initial_conds, times)
        theta = solution[:, 0]
        omega = solution[:, 1]
        gamma = - GRAVITY / LENGTH * np.sin(theta)

        # Create data frame and save csv file
        df = pd.DataFrame()
        df["Time"] = np.round(times, 5)
        df["Angle"] = np.round(theta, 5)
        df["AngularVelocity"] = np.round(omega, 5)
        df["AngularAcceleration"] = np.round(gamma, 5)
        df["TangentialVelocity"] = np.round(LENGTH * df["AngularVelocity"], 5)
        df["RadialAcceleration"] = np.round(
            - LENGTH * df["AngularVelocity"]**2, 5)
        df["TangentialAcceleration"] = np.round(
            LENGTH * df["AngularAcceleration"], 5)
        df["xPosition"] = np.round(LENGTH * np.cos(df["Angle"]), 5)
        df["yPosition"] = np.round(LENGTH * np.sin(df["Angle"]), 5)
        df["zPosition"] = np.zeros(times.shape)
        df["Weight"] = np.round(MASS * GRAVITY, 5)
        df["Tension"] = np.round(MASS * GRAVITY * np.cos(theta)
                                 + MASS * LENGTH * omega**2, 5)

        df.to_csv("data/pendulum.csv", index=False)


class Pendulum:
    # TODO: Add documentation.
    def __init__(self, x0: float, y0: float, z0: float):
        colors = Colors()

        self.radius = 0.1
        self.mass = 1
        self.momentum = vp.vec(0, 0, 0)
        self.color = colors.blue
        self.trail_color = colors.blue

        self.ball = vp.sphere(pos=Vector(x0, y0, z0),
                              make_trail=True,
                              retain=15,
                              trail_radius=0.01)
        self.ball.radius = self.radius
        self.ball.masss = self.mass
        self.ball.momentum = self.momentum
        self.ball.color = self.color
        self.ball.trail_color = self.color

        self.rope = vp.curve([Vector(0, 0, 0), self.ball.pos],
                             color=self.color,
                             radius=0.01)

    def update_pos(self, x: float, y: float, z: float) -> None:
        # TODO: Add documentation.
        self.ball.pos = Vector(x, y, z)
        self.rope.modify(1, Vector(x, y, z))


class Grid:
    # TODO: Add documentation.
    def __init__(self):
        colors = Colors()

        x_num = int(np.round((XMAX - XMIN) / STEP + 1, 0))
        y_num = int(np.round((YMAX - YMIN) / STEP + 1))

        for y in np.linspace(YMIN, YMAX, y_num):
            vp.curve(Vector(XMIN, y, 0), Vector(XMAX, y, 0),
                     color=colors.gray, radius=0.001)
        for x in np.linspace(XMIN, XMAX, x_num):
            vp.curve(Vector(x, YMIN, 0), Vector(x, YMAX, 0),
                     color=colors.gray, radius=0.001)


class CartesianFrame:
    # TODO: Add documentation.
    def __init__(self):
        colors = Colors()

        self.x_axis = vp.arrow(
            pos=Vector(0, 0, 0),
            axis=Displacement(1, 0, 0), color=colors.green,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.y_axis = vp.arrow(
            pos=Vector(0, 0, 0),
            axis=Displacement(0, 1, 0), color=colors.green,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.z_axis = vp.arrow(
            pos=Vector(0, 0, 0),
            axis=Displacement(0, 0, 1), color=colors.green,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        vp.label(pos=self.x_axis.pos + self.x_axis.axis, text='x', height=20,
                 box=False, color=colors.green)
        vp.label(pos=self.y_axis.pos + self.y_axis.axis, text='y', height=20,
                 box=False, color=colors.green)
        vp.label(pos=self.z_axis.pos + self.z_axis.axis, text='z', height=20,
                 box=False, color=colors.green)


class Forces:
    # TODO: Add documentation.
    def __init__(self, scale: float,
                 x0: float, y0: float, z0: float,
                 tension: float, weight: float):
        colors = Colors()

        self.scale = scale

        # Set up the tension
        self.tension = vp.arrow(
            pos=Vector(x0, y0, z0),
            axis=-self.scale * tension * Displacement(x0, y0, z0).norm(),
            color=colors.yellow,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.tension_label = vp.label(
            pos=self.tension.pos + self.tension.axis,
            text="<b>T</b>", height=20,
            box=False, color=colors.yellow)

        # Set up the weight
        self.weight = vp.arrow(
            pos=Vector(x0, y0, z0),
            axis=self.scale * weight * Displacement(1, 0, 0).norm(),
            color=colors.yellow,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.weight_label = vp.label(
            pos=self.weight.pos + self.weight.axis,
            text="<i>m</i><b>g</b>", height=20,
            box=False, color=colors.yellow)

    def update_pos(self, x: float, y: float, z: float,
                   tension: float, weight: float):
        # TODO: Add documentation.
        # Update the tension
        self.tension.pos = Vector(x, y, z)
        self.tension.axis = -self.scale * tension \
            * Displacement(x, y, z).norm()
        self.tension_label.pos = self.tension.pos + self.tension.axis

        # Update the weight
        self.weight.pos = Vector(x, y, z)
        self.weight_label.pos = self.weight.pos + self.weight.axis


class Dynamics:
    # TODO: Add documentation.
    def __init__(self, scale: float,
                 x0: float, y0: float, z0: float,
                 radial_acc: float, tangential_acc: float,
                 tangential_vel: float):
        colors = Colors()

        self.scale = scale

        # Set up the radial acceleration vector and label
        self.radial_acc = vp.arrow(
            pos=Vector(x0, y0, z0),
            axis=self.scale * radial_acc * Displacement(x0, y0, z0).norm(),
            color=colors.red,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.radial_acc_label = vp.label(
            pos=self.radial_acc.pos + self.radial_acc.axis,
            text="<b>a</b><sub>r</sub>", height=20,
            box=False, color=colors.red)

        # Set up the tangential acceleration vector and label
        self.tangential_acc = vp.arrow(
            pos=Vector(x0, y0, z0),
            axis=self.scale
            * tangential_acc
            * Displacement(x0, y0, z0).norm().rotate(vp.radians(90),
                                                     Displacement(0, 0, 1)),
            color=colors.red,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.tangential_acc_label = vp.label(
            pos=self.tangential_acc.pos + self.tangential_acc.axis,
            text="<b>a</b><sub>t</sub>", height=20,
            box=False, color=colors.red)

        # Set up the tangetial acceleration vector and label
        self.tangential_vel = vp.arrow(
            pos=Vector(x0, y0, z0),
            axis=self.scale
            * tangential_vel
            * Displacement(x0, y0, z0).norm().rotate(vp.radians(90),
                                                     Displacement(0, 0, 1)),
            color=colors.red,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.tangential_vel_label = vp.label(
            pos=self.tangential_vel.pos + self.tangential_vel.axis,
            text="<b>v</b><sub>t</sub>", height=20,
            box=False, color=colors.red)

    def update_pos(self, x: float, y: float, z: float,
                   radial_acc: float, tangential_acc: float,
                   tangential_vel: float):
        # TODO: Add documentation.
        # Update the radial acceleration
        self.radial_acc.pos = Vector(x, y, z)
        self.radial_acc.axis = self.scale * radial_acc \
            * Displacement(x, y, z).norm()
        self.radial_acc_label.pos = self.radial_acc.pos + self.radial_acc.axis

        # Update the tangential acceleration
        self.tangential_acc.pos = Vector(x, y, z)
        self.tangential_acc.axis = self.scale * tangential_acc * \
            Displacement(x, y, z).norm().rotate(vp.radians(90),
                                                Displacement(0, 0, 1))
        self.tangential_acc_label.pos = self.tangential_acc.pos + \
            self.tangential_acc.axis

        # Update the tangential velocity
        self.tangential_vel.pos = Vector(x, y, z)
        self.tangential_vel.axis = self.scale * tangential_vel * \
            Displacement(x, y, z).norm().rotate(vp.radians(90),
                                                Displacement(0, 0, 1))
        self.tangential_vel_label.pos = self.tangential_vel.pos + \
            self.tangential_vel.axis


class CylindricalFrame:
    # TODO: Add documentation.
    def __init__(self, scale: float, x0: float, y0: float, z0: float):
        colors = Colors()

        self.scale = scale

        self.r_axis = vp.arrow(
            pos=Vector(0, 0, 0),
            axis=self.scale * Displacement(x0, y0, 0).norm(),
            color=colors.purple,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.t_axis = vp.arrow(
            pos=Vector(0, 0, 0),
            axis=self.r_axis.axis.rotate(vp.radians(90),
                                         Displacement(0, 0, 1)),
            color=colors.purple,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.z_axis = vp.arrow(
            pos=Vector(0, 0, 0),
            axis=self.scale * Displacement(0, 0, 1),
            color=colors.purple,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.r_label = vp.label(
            pos=self.r_axis.pos + self.r_axis.axis,
            text='r', height=20,
            box=False, color=colors.purple)
        self.t_label = vp.label(
            pos=self.t_axis.pos + self.t_axis.axis,
            text='θ', height=20,
            box=False, color=colors.purple)
        self.z_label = vp.label(
            pos=self.z_axis.pos + self.z_axis.axis,
            text='z', height=20,
            box=False, color=colors.purple)

    def update_pos(self, x: float, y: float, z: float):
        # TODO: Add documentation.
        self.r_axis.axis = self.scale * Displacement(x, y, z).norm()
        self.r_label.pos = self.r_axis.pos + self.r_axis.axis

        self.t_axis.axis = self.r_axis.axis.rotate(vp.radians(90),
                                                   Displacement(0, 0, 1))
        self.t_label.pos = self.t_axis.pos + self.t_axis.axis


def create_scene(forces: bool = False, dynamics: bool = False):
    # TODO: Add documentation.
    colors = Colors()

    vp.scene.width = SCENE_WIDTH
    vp.scene.height = SCENE_HEIGHT
    vp.scene.background = colors.background
    vp.scene.caption = """
    Para rotar la cámara, arrastra con el botón derecho del mouse o arrastra
    mientras mantienes presionada la tecla Ctrl. Para hacer zoom, usa la rueda
    de desplazamiento. Para desplazarte hacia la izquierda/derecha y hacia
    arriba/abajo, arrastra mientras mantienes presionada la tecla Shift. En
    una pantalla táctil: pellizca/extiende para hacer zoom, desliza o rota
    con dos dedos.
    """
    vp.scene.range = SCENE_RANGE
    vp.scene.forward = vp.vec(-1, -1, -1)
    vp.distant_light(direction=vp.vec(-1, -1, -1), color=vp.color.white)

    # Load data
    df = pd.read_csv("data/pendulum.csv")

    x0 = df.loc[0, "xPosition"]
    y0 = df.loc[0, "yPosition"]
    z0 = df.loc[0, "zPosition"]

    # Coordinate systems
    Grid()
    CartesianFrame()
    cylindrical_frame = CylindricalFrame(CYLINDRICAL_FRAME_SCALE, x0, y0, z0)

    # Draw pendulum
    pendulum = Pendulum(x0, y0, z0)

    # Forces
    if forces:
        forces = Forces(FORCES_SCALE,
                        x0, y0, z0,
                        tension=df.loc[0, "Tension"],
                        weight=df.loc[0, "Weight"])

    if dynamics:
        dynamics = Dynamics(DYNAMICS_SCALE,
                            x0, y0, z0,
                            radial_acc=df.loc[0, "RadialAcceleration"],
                            tangential_acc=df.loc[0, "TangentialAcceleration"],
                            tangential_vel=df.loc[0, "TangentialVelocity"])

    time.sleep(5.0)

    i = 0
    while True:
        vp.rate(FRAMERATE)

        x = df.loc[i, "xPosition"]
        y = df.loc[i, "yPosition"]
        z = df.loc[i, "zPosition"]
        radial_acc = df.loc[i, "RadialAcceleration"]
        tangential_acc = df.loc[i, "TangentialAcceleration"]
        tangential_vel = df.loc[i, "TangentialVelocity"]
        tension = df.loc[i, "Tension"]
        weight = df.loc[i, "Weight"]

        pendulum.update_pos(x, y, z)
        cylindrical_frame.update_pos(x, y, z)

        if forces:
            # Update the force vectors
            forces.update_pos(x, y, z, tension, weight)

        if dynamics:
            # Update the position of the dynamic vectors
            dynamics.update_pos(x, y, z,
                                radial_acc, tangential_acc, tangential_vel)

        i += 1
        if i >= len(df):
            i = 0


if __name__ == "__main__":
    # Solve the dynamics
    solver = Solver()
    solver.solve(FINAL_TIME, TIME_STEP, THETA0, OMEGA0)

    # Create scene
    create_scene(dynamics=True, forces=True)
