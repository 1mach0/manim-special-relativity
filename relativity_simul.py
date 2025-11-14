from manim import *
import numpy as np

speed_of_light = 1.0

# stole from manim physics
class Charge(VGroup):
    def __init__(
        self,
        magnitude: float = 1,
        point: np.ndarray = ORIGIN,
        add_glow: bool = True,
        **kwargs,
    ) -> None:
        """An electrostatic charge object to produce an :class:`~ElectricField`.

        Parameters
        ----------
        magnitude
            The strength of the electrostatic charge.
        point
            The position of the charge.
        add_glow
            Whether to add a glowing effect. Adds rings of
            varying opacities to simulate glowing effect.
        kwargs
            Additional parameters to be passed to ``VGroup``.
        """
        VGroup.__init__(self, **kwargs)
        self.magnitude = magnitude
        self.point = point
        self.radius = (abs(magnitude) * 0.4 if abs(magnitude) < 2 else 0.8) * 0.3

        if magnitude > 0:
            label = VGroup(
                Rectangle(width=0.32 * 1.1, height=0.006 * 1.1).set_z_index(1),
                Rectangle(width=0.006 * 1.1, height=0.32 * 1.1).set_z_index(1),
            )
            color = RED
            layer_colors = [RED_D, RED_A]
            layer_radius = 4
        else:
            label = Rectangle(width=0.27, height=0.003)
            color = BLUE
            layer_colors = ["#3399FF", "#66B2FF"]
            layer_radius = 2

        if add_glow:  # use many arcs to simulate glowing
            layer_num = 80
            color_list = color_gradient(layer_colors, layer_num)
            opacity_func = lambda t: 1500 * (1 - abs(t - 0.009) ** 0.0001)
            rate_func = lambda t: t**2

            for i in range(layer_num):
                self.add(
                    Arc(
                        radius=layer_radius * rate_func((0.5 + i) / layer_num),
                        angle=TAU,
                        color=color_list[i],
                        stroke_width=101
                        * (rate_func((i + 1) / layer_num) - rate_func(i / layer_num))
                        * layer_radius,
                        stroke_opacity=opacity_func(rate_func(i / layer_num)),
                    ).shift(point)
                )

        self.add(Dot(point=self.point, radius=self.radius, color=color))
        self.add(label.scale(self.radius / 0.3).shift(point))
        for mob in self:
            mob.set_z_index(1)

class RelativisticCharge(Charge):
    def __init__(self, magnitude=1.0, position=ORIGIN, velocity=np.array([0.2, 0, 0]), add_glow=True):
        super().__init__(magnitude, position, add_glow)
        self.velocity = np.array(velocity)

class RelativisticEMField(ArrowVectorField):
    def __init__(self, *charges: Charge, **kwargs):
        self.charges = charges
        super().__init__(lambda p: self.electric_field_func(
            p,
            [c.get_center() for c in charges],
            [c.magnitude for c in charges],
            [c.velocity for c in charges]
        ), **kwargs)

    def electric_field_func(self, p, positions, magnitudes, velocities):
        field = np.zeros(3)

        for p0, q, v in zip(positions, magnitudes, velocities):
            R = p - p0
            R_norm = np.linalg.norm(R)
            if R_norm < 0.1:
                continue

            v_norm = np.linalg.norm(v)
            if v_norm < 1e-6:
                field += q * R / (R_norm ** 3)
                continue

            beta = v_norm / speed_of_light
            if beta >= 0.999999:
                beta = 0.999999
            gamma = 1.0 / np.sqrt(1 - beta ** 2)

            v_hat = v / v_norm

            R_parallel = np.dot(R, v_hat) * v_hat
            R_perp = R - R_parallel

            R_perp_sq = np.dot(R_perp, R_perp)
            R_par_sq = np.dot(R_parallel, R_parallel)

            denom = (R_perp_sq + (gamma ** 2) * R_par_sq) ** 1.5

            E = q * ((1 - beta ** 2) * R_perp + R_parallel) / denom

            field += E

        return field

class RelativisticMagneticField(RelativisticEMField):
    def magnetic_field_func(self, p, positions, magnitudes, velocities):
        b_field = np.zeros(3)

        for p0, q, v in zip(positions, magnitudes, velocities):
            R = p - p0
            R_norm = np.linalg.norm(R)
            if R_norm < 0.1:
                continue

            v_norm = np.linalg.norm(v)
            if v_norm < 1e-6:
                continue

            beta = v_norm / speed_of_light
            if beta >= 0.999999:
                beta = 0.999999
            gamma = 1.0 / np.sqrt(1 - beta**2)
            v_hat = v / v_norm

            R_parallel = np.dot(R, v_hat) * v_hat
            R_perp = R - R_parallel

            R_perp_sq = np.dot(R_perp, R_perp)
            R_par_sq = np.dot(R_parallel, R_parallel)
            denom = (R_perp_sq + (gamma**2) * R_par_sq)**1.5

            E = q * ((1 - beta**2) * R_perp + R_parallel) / denom

            b_field += np.cross(v, E) / (speed_of_light**2)

        return b_field


class ElectricFieldExample(ThreeDScene):
    def construct(self):
        charge = RelativisticCharge(1, np.array([0,0,0]), np.array([0,0,0.0]))

        def E_func(p):
            return RelativisticEMField().electric_field_func(
                p,
                [charge.get_center()],
                [charge.magnitude],
                [charge.velocity]
            )

        def B_func(p):
            return RelativisticMagneticField().magnetic_field_func(
                p,
                [charge.get_center()],
                [charge.magnitude],
                [charge.velocity]
            )

        EMfield = ArrowVectorField(
            E_func,
            x_range=[-3, 3, 0.6],
            y_range=[-3, 3, 0.6],
            z_range=[0, 0, 1],
            color=BLUE
        )

        Magfield = ArrowVectorField(
            B_func,
            x_range=[-3, 3, 0.6],
            y_range=[-3, 3, 0.6],
            z_range=[0, 0, 1],
            color=YELLOW
        )

        self.add(EMfield, Magfield, charge)

        def update_velocity(m, dt):
            m.velocity[2] = min(m.velocity[2] + dt * 0.15, 0.99)

        charge.add_updater(update_velocity)

        EMfield.add_updater(
            lambda m: m.become(
                ArrowVectorField(
                    E_func,
                    x_range=[-3, 3, 0.6],
                    y_range=[-3, 3, 0.6],
                    z_range=[0, 0, 1],
                    color=BLUE
                )
            )
        )

        Magfield.add_updater(
            lambda m: m.become(
                ArrowVectorField(
                    B_func,
                    x_range=[-3, 3, 0.6],
                    y_range=[-3, 3, 0.6],
                    z_range=[0, 0, 1],
                    color=YELLOW
                )
            )
        )

        self.wait(4)
