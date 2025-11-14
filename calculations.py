import numpy as np
import matplotlib.pyplot as plt

c = 1.0

class ElectricField:
    def __init__(self, charge, velocity, position=np.zeros(3)):
        self.charge = charge
        self.velocity = np.asarray(velocity, dtype=float)
        self.position = np.asarray(position, dtype=float)

        v = np.linalg.norm(self.velocity)
        if v >= c:
            raise ValueError("Velocity must be less than speed of light.")

        self.gamma = 1.0 / np.sqrt(1.0 - (v / c)**2)

    def field(self, observation_points):
        obs = np.asarray(observation_points, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, 3)

        # geometry
        R = obs - self.position
        v = self.velocity
        vnorm = np.linalg.norm(v)

        if vnorm < 1e-16:
            vhat = np.zeros(3)
        else:
            vhat = v / vnorm

        if vnorm < 1e-16:
            R_par = np.zeros_like(R)
            R_perp = R.copy()
        else:
            proj = np.dot(R, vhat)[:, None] * vhat[None, :]
            R_par = proj
            R_perp = R - proj

        R_prime = self.gamma * R_par + R_perp
        rprime_mag = np.linalg.norm(R_prime, axis=1) + 1e-12

        E_prime = (self.charge / (rprime_mag**3))[:, None] * R_prime

        if vnorm < 1e-16:
            E_prime_par = np.zeros_like(E_prime)
            E_prime_perp = E_prime.copy()
        else:
            eproj = np.dot(E_prime, vhat)[:, None] * vhat[None, :]
            E_prime_par = eproj
            E_prime_perp = E_prime - eproj

        E_lab = E_prime_par + self.gamma * E_prime_perp

        B_lab = -self.gamma * np.cross(v[None, :], E_prime)

        return E_lab, B_lab

Nx, Ny, Nz = 20, 20, 20

x = np.linspace(-5, 5, Nx)
y = np.linspace(-5, 5, Ny)
z = np.linspace(-5, 5, Nz)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

obs_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

charge = ElectricField(
    charge=1.0,
    velocity=np.array([0.7, 0, 0]),
    position=np.array([0.0, 0.0, 0.0])
)

E, B = charge.field(obs_points)

E_field = E.reshape(Nx, Ny, Nz, 3)
B_field = B.reshape(Nx, Ny, Nz, 3)

x_index = Nx // 2

Y = yy[x_index, :, :]
Z = zz[x_index, :, :]

By = B_field[x_index, :, :, 1]
Bz = B_field[x_index, :, :, 2]

