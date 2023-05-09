import vpython as vp
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import time

from colors import Colors

# Device properties
MASS: float = 1.0  # kg
RADIUS: float = 1.5  # m
GRAVITY: float = 9.8  # m/s^2
THETA0: float = np.radians(1.0)  # radians
OMEGA0: float = 0.0  # 1/s

# Scene properties
DEVICE_HEIGHT: float = 1.0  # m
TIME_STEP = 0.001  # s
FINAL_TIME: float = 5.0  # s
FRAMERATE = int(1 / TIME_STEP)
SCENE_WIDTH: int = 600  # px
SCENE_HEIGHT: int = 600  # px
SCENE_RANGE: float = 2.0  # m
CYLINDRICAL_FRAME_SCALE: float = 0.5
BALL_RADIUS: float = 0.1  # m
DYNAMICS_SCALE: float = 0.15
FORCES_SCALE: float = 0.05
FLOOR_SIZE: float = 2.0  # m
FLOOR_HEIGHT: float = 0.05  # m
FLOOR_OPACITY: float = 0.1

# Grid configuration
XMIN: float = -1.8
XMAX: float = 1.8
YMIN: float = -1.8
YMAX: float = 1.8
STEP: float = 0.2


class Vector(vp.vector):
    """
    A class to manage vectors in the coordinate system of the current
    device.
    """
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y, z)
        self._transform()

    def _transform(self) -> None:
        """
        This method transforms the coordinates from the local system to the
        global system of VPython.
        """
        x = self.x
        y = self.y
        z = self.z

        self.x = y
        self.y = x
        self.z = -z


class Displacement(vp.vector):
    """
    A class to manage displacement vectors in the coordinate system of the
    current device.
    """
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y, z)
        self._transform()

    def _transform(self) -> None:
        """
        This method transforms the coordinates from the local system to the
        global system of VPython.
        """
        x = self.x
        y = self.y
        z = self.z

        self.x = y
        self.y = x
        self.z = -z


class Solver:
    """
    A class to manage the solver of the differential equations that define
    this problem.
    """
    def _equations(self, initial_conds: list, times: np.ndarray) -> None:
        """
        This method defines the differential equations.

        Parameters
        ----------
        initial_conds : list
            A list where the first element is the initial angle in radians
            and the second element is the initial angular velocity in radians
            per second.
        times : np.ndarray
            An array with the times (in seconds) to analyze.
        """
        theta, omega = initial_conds
        f = [omega, GRAVITY / RADIUS * np.sin(theta)]
        return f

    def solve(self, tf: float, dt: float, theta0: float, omega0: float):
        """
        This method solves the differential equations given the inputs, and
        creates and saves a data frame with all the relevant physical
        properties.

        Parameters
        ----------
        tf : float
            The last time in seconds.
        dt : float
            The time step in seconds.
        theta0 : float
            The initial angle in radians.
        omega0 : float
            The initial angular velocity in radians per second.
        """
        initial_conds = [theta0, omega0]
        times = np.arange(0, tf, dt)

        # Solve the differential equations
        solution = odeint(self._equations, initial_conds, times)
        theta = solution[:, 0]
        omega = solution[:, 1]
        gamma = GRAVITY / RADIUS * np.sin(theta)

        # Create data frame and save csv file
        df = pd.DataFrame()
        df["Time"] = np.round(times, 5)
        df["Angle"] = np.round(theta, 5)
        df["AngularVelocity"] = np.round(omega, 5)
        df["AngularAcceleration"] = np.round(gamma, 5)
        df["TangentialVelocity"] = np.round(RADIUS * df["AngularVelocity"], 5)
        df["RadialAcceleration"] = np.round(
            - RADIUS * df["AngularVelocity"]**2, 5)
        df["TangentialAcceleration"] = np.round(
            RADIUS * df["AngularAcceleration"], 5)
        df["xPosition"] = np.round(RADIUS * np.cos(df["Angle"]), 5)
        df["yPosition"] = np.round(RADIUS * np.sin(df["Angle"]), 5)
        df["zPosition"] = np.zeros(times.shape)
        df["Weight"] = np.round(MASS * GRAVITY, 5)
        df["Normal"] = np.round(MASS * GRAVITY * np.cos(theta)
                                - MASS * RADIUS * omega**2, 5)

        # Remove all values outside the range of the problem
        mask = df['Angle'] >= np.pi / 2
        first_index = mask.cumsum().eq(1).idxmax()
        df = df.loc[:first_index]

        df.to_csv("data/semi-sphere.csv", index=False)


class Ball:
    """
    A class to manage the 3D model of the ball.
    """
    def __init__(self, x0: float, y0: float, z0: float):
        """
        The constructor method.

        Parameters
        ----------
        x0 : float
            The initial position in the x-axis in meters.
        y0 : float
            The initial position in the y-axis in meters.
        z0 : float
            The initial position in the z-axis in meters.
        """
        colors = Colors()

        self.radius = BALL_RADIUS
        self.mass = 1
        self.momentum = vp.vec(0, 0, 0)
        self.color = colors.blue
        self.trail_color = colors.blue

        self.ball = vp.sphere(pos=Vector(x0, y0, z0),
                              make_trail=False)
        self.ball.radius = self.radius
        self.ball.masss = self.mass
        self.ball.momentum = self.momentum
        self.ball.color = self.color
        self.ball.trail_color = self.color

    def update_pos(self, x: float, y: float, z: float) -> None:
        """
        This method updates the position of the ball to the given values.

        Parameters
        ----------
        x : float
            The position in the x-axis in meters.
        y : float
            The position in the y-axis in meters.
        z : float
            The position in the z-axis in meters.
        """
        self.ball.pos = Vector(x, y, z)


class Grid:
    """
    A class to manage the construction of a grid in the model.
    """
    def __init__(self):
        colors = Colors()

        x_num = int(np.round((XMAX - XMIN) / STEP + 1))
        y_num = int(np.round((YMAX - YMIN) / STEP + 1))

        for y in np.linspace(YMIN, YMAX, y_num):
            vp.curve(Vector(XMIN, y, 0), Vector(XMAX, y, 0),
                     color=colors.gray, radius=0.001)
        for x in np.linspace(XMIN, XMAX, x_num):
            vp.curve(Vector(x, YMIN, 0), Vector(x, YMAX, 0),
                     color=colors.gray, radius=0.001)


class Floor:
    """
    A class to manage the floor of the 3D model.
    """
    def __init__(self):
        colors = Colors()

        vp.box(pos=Vector(0, 0, 0),
               length=2 * FLOOR_SIZE,
               height=FLOOR_HEIGHT,
               width=2 * FLOOR_SIZE,
               color=colors.gray,
               opacity=FLOOR_OPACITY)


class CartesianFrame:
    """
    A class to manage the constructor of the 3D model for the cartesian frame
    of reference.
    """
    def __init__(self):
        """
        The constructor method.
        """
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
    """
    A class to manage the construction of the 3D models of the forces.
    """
    def __init__(self, scale: float,
                 x0: float, y0: float, z0: float,
                 normal0: float, weight0: float):
        """
        The constructor method

        Parameters
        ----------
        scale : float
            A multiplicative scale factor to apply to the forces.
        x0 : float
            The initial position in the x-axis in meters.
        y0 : float
            The initial position in the y-axis in meters.
        z0 : float
            The initial position in the z-axis in meters.
        normal0 : float
            The initial value of the normal in newtons.
        weight0 : float
            The initial value of the weight in newtons.
        """
        colors = Colors()

        self.scale = scale

        # Set up the tension
        self.normal = vp.arrow(
            pos=Vector(x0, y0, z0),
            axis=self.scale * normal0 * Displacement(x0, y0, z0).norm(),
            color=colors.yellow,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.normal_label = vp.label(
            pos=self.normal.pos + self.normal.axis,
            text="<b>N</b>", height=20,
            box=False, color=colors.yellow)

        # Set up the weight
        self.weight = vp.arrow(
            pos=Vector(x0, y0, z0),
            axis=self.scale * weight0 * Displacement(-1, 0, 0).norm(),
            color=colors.yellow,
            round=True, shaftwidth=0.02, headwidth=0.04,
            headlength=0.05)
        self.weight_label = vp.label(
            pos=self.weight.pos + self.weight.axis,
            text="<i>m</i><b>g</b>", height=20,
            box=False, color=colors.yellow)

    def update_pos(self, x: float, y: float, z: float,
                   tension: float, weight: float):
        """
        This method update the 3D model of the forces to the given values.

        Parameters
        ----------
        x : float
            The position in the x-axis in meters.
        y : float
            The position in the y-axis in meters.
        z : float
            The position in the z-axis in meters.
        normal : float
            The value of the tension in newtons.
        weight : float
            The value of the tension in newtons.
        """
        # Update the tension
        self.normal.pos = Vector(x, y, z)
        self.normal.axis = self.scale * tension * Displacement(x, y, z).norm()
        self.normal_label.pos = self.normal.pos + self.normal.axis

        # Update the weight
        self.weight.pos = Vector(x, y, z)
        self.weight_label.pos = self.weight.pos + self.weight.axis


# TODO: Add the velocity and acceleration vectors
# class Dynamics:
#     """
#     A class to manage the 3D models of the velocity and acceleration.
#     """
#     def __init__(self, scale: float,
#                  x0: float, y0: float, z0: float,
#                  radial_acc0: float, tangential_acc0: float,
#                  tangential_vel0: float):
#         """
#         The constructor method.

#         Parameters
#         ----------
#         scale : float
#             A multiplicative scale factor to apply to the forces.
#         x0 : float
#             The initial position in the x-axis in meters.
#         y0 : float
#             The initial position in the y-axis in meters.
#         z0 : float
#             The initial position in the z-axis in meters.
#         radial_acc0 : float
#             The initial value of the radial acceleration in meters per
#             second squared.
#         tangential_acc0 : float
#             The initial value of the tangential acceleration in meters
#             per second squared.
#         tangential_vel0 : float
#             The initial value of the tangential velocity in meters
#             per second.
#         """
#         colors = Colors()

#         self.scale = scale

#         # Set up the radial acceleration vector and label
#         self.radial_acc = vp.arrow(
#             pos=Vector(x0, y0, z0),
#             axis=self.scale * radial_acc0 * Displacement(x0, y0, z0).norm(),
#             color=colors.red,
#             round=True, shaftwidth=0.02, headwidth=0.04,
#             headlength=0.05)
#         self.radial_acc_label = vp.label(
#             pos=self.radial_acc.pos + self.radial_acc.axis,
#             text="<b>a</b><sub>r</sub>", height=20,
#             box=False, color=colors.red)

#         # Set up the tangential acceleration vector and label
#         self.tangential_acc = vp.arrow(
#             pos=Vector(x0, y0, z0),
#             axis=self.scale
#             * tangential_acc0
#             * Displacement(x0, y0, z0).norm().rotate(vp.radians(90),
#                                                      Displacement(0, 0, 1)),
#             color=colors.red,
#             round=True, shaftwidth=0.02, headwidth=0.04,
#             headlength=0.05)
#         self.tangential_acc_label = vp.label(
#             pos=self.tangential_acc.pos + self.tangential_acc.axis,
#             text="<b>a</b><sub>t</sub>", height=20,
#             box=False, color=colors.red)

#         # Set up the tangetial acceleration vector and label
#         self.tangential_vel = vp.arrow(
#             pos=Vector(x0, y0, z0),
#             axis=self.scale
#             * tangential_vel0
#             * Displacement(x0, y0, z0).norm().rotate(vp.radians(90),
#                                                      Displacement(0, 0, 1)),
#             color=colors.red,
#             round=True, shaftwidth=0.02, headwidth=0.04,
#             headlength=0.05)
#         self.tangential_vel_label = vp.label(
#             pos=self.tangential_vel.pos + self.tangential_vel.axis,
#             text="<b>v</b><sub>t</sub>", height=20,
#             box=False, color=colors.red)

#     def update_pos(self, x: float, y: float, z: float,
#                    radial_acc: float, tangential_acc: float,
#                    tangential_vel: float):
#         """
#         This method updates the 3D model of the forces.

#         Parameters
#         ----------
#         x : float
#             The position in the x-axis in meters.
#         y : float
#             The position in the y-axis in meters.
#         z : float
#             The position in the z-axis in meters.
#         radial_acc : float
#             The value of the radial acceleration in meters per
#             second squared.
#         tangential_acc : float
#             The value of the tangential acceleration in meters
#             per second squared.
#         tangential_vel : float
#             The value of the tangential velocity in meters
#             per second.
#         """
#         # Update the radial acceleration
#         self.radial_acc.pos = Vector(x, y, z)
#         self.radial_acc.axis = self.scale * radial_acc \
#             * Displacement(x, y, z).norm()
#         self.radial_acc_label.pos = self.radial_acc.pos + self.radial_acc.axis

#         # Update the tangential acceleration
#         self.tangential_acc.pos = Vector(x, y, z)
#         self.tangential_acc.axis = self.scale * tangential_acc * \
#             Displacement(x, y, z).norm().rotate(vp.radians(90),
#                                                 Displacement(0, 0, 1))
#         self.tangential_acc_label.pos = self.tangential_acc.pos + \
#             self.tangential_acc.axis

#         # Update the tangential velocity
#         self.tangential_vel.pos = Vector(x, y, z)
#         self.tangential_vel.axis = self.scale * tangential_vel * \
#             Displacement(x, y, z).norm().rotate(vp.radians(90),
#                                                 Displacement(0, 0, 1))
#         self.tangential_vel_label.pos = self.tangential_vel.pos + \
#             self.tangential_vel.axis


class CylindricalFrame:
    """
    A class to manage the constructor of the 3D model for the cylindrical frame
    of reference.
    """
    def __init__(self, scale: float, x0: float, y0: float, z0: float):
        """
        The constructor method.

        Parameters
        ----------
        scale : float
            A multiplicative scale factor to apply to the versors.
        x0 : float
            The initial position in the x-axis in meters.
        y0 : float
            The initial position in the y-axis in meters.
        z0 : float
            The initial position in the z-axis in meters.
        """
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
        """
        This method updates the 3D model of the frame.

        Parameters
        ----------
        x : float
            The position in the x-axis in meters.
        y : float
            The position in the y-axis in meters.
        z : float
            The position in the z-axis in meters.
        """
        self.r_axis.axis = self.scale * Displacement(x, y, z).norm()
        self.r_label.pos = self.r_axis.pos + self.r_axis.axis

        self.t_axis.axis = self.r_axis.axis.rotate(vp.radians(90),
                                                   Displacement(0, 0, 1))
        self.t_label.pos = self.t_axis.pos + self.t_axis.axis


def create_scene(forces: bool = False, dynamics: bool = False):
    """
    This method creates the 3D scene.

    Parameters
    ----------
    forces : bool, optional
        If True, show the forces in the 3D model, by default False.
    dynamics : bool, optional
        If True, show the velocity and acceleration vectors in the 3D model,
        by default False.
    """
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
    df = pd.read_csv("data/semi-sphere.csv")

    x0 = df.loc[0, "xPosition"]
    y0 = df.loc[0, "yPosition"]
    z0 = df.loc[0, "zPosition"]

    # Coordinate systems
    # Grid()
    CartesianFrame()
    cylindrical_frame = CylindricalFrame(CYLINDRICAL_FRAME_SCALE, x0, y0, z0)

    # Draw ball support
    # TODO: Change the support to a curved surface using extrusion
    angles = np.linspace(-np.pi / 2, np.pi / 2, 200)
    support = vp.curve(color=colors.blue)
    for angle in angles:
        point = RADIUS * Vector(np.cos(angle), np.sin(angle), 0)
        support.append(point)

    # Draw pendulum
    ball = Ball(x0, y0, z0)

    # Draw floor
    Floor()

    # Forces
    if forces:
        forces = Forces(
            FORCES_SCALE,
            x0, y0, z0,
            normal0=df.loc[0, "Normal"],
            weight0=df.loc[0, "Weight"])

    if dynamics:
        pass
        # dynamics = Dynamics(
        #     DYNAMICS_SCALE,
        #     x0, y0, z0,
        #     radial_acc0=df.loc[0, "RadialAcceleration"],
        #     tangential_acc0=df.loc[0, "TangentialAcceleration"],
        #     tangential_vel0=df.loc[0, "TangentialVelocity"])

    time.sleep(5.0)

    i = 0
    while True:
        vp.rate(FRAMERATE)

        x = df.loc[i, "xPosition"]
        y = df.loc[i, "yPosition"]
        z = df.loc[i, "zPosition"]
        # radial_acc = df.loc[i, "RadialAcceleration"]
        # tangential_acc = df.loc[i, "TangentialAcceleration"]
        # tangential_vel = df.loc[i, "TangentialVelocity"]
        normal = df.loc[i, "Normal"]
        weight = df.loc[i, "Weight"]

        ball.update_pos(x, y, z)
        cylindrical_frame.update_pos(x, y, z)

        if forces:
            # Update the force vectors
            forces.update_pos(x, y, z, normal, weight)

        if dynamics:
            pass
            # # Update the position of the dynamic vectors
            # dynamics.update_pos(x, y, z,
            #                     radial_acc, tangential_acc, tangential_vel)

        i += 1
        if i >= len(df):
            i = 0
            time.sleep(3)


if __name__ == "__main__":
    # Solve the dynamics
    solver = Solver()
    solver.solve(FINAL_TIME, TIME_STEP, THETA0, OMEGA0)

    # Create scene
    create_scene(dynamics=True, forces=True)
