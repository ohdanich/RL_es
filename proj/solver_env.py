import sys

import numpy as np
from scipy.integrate import ode

from utils import plot_control


class Model:
    def __init__(self, r_mv=0.5, m_mv=1.2, k_m=10, P_m=500,
                 k_ts=0.02, k_fr_r=0.6, k_fr_b=0.01, k_bh=165, m_pr=10,
                 J_psi=10, m_r=2, l_r=3, s_p=0.1, m_p=4,
                 d_cm=0.2, k_r=0.01, k_gyr=2):
        self.r_mv = r_mv  # Радиус мотор-колеса, м
        self.m_mv = m_mv  # Масса мотор-колеса, кг
        self.J_mv = 0.666 * r_mv * r_mv * m_mv  # Момент инерции мотор-колеса
        self.k_m = k_m  # Коэффициент электромотора
        self.P_m = P_m  # Мощность электромотора, Вт
        self.k_ts = k_ts  # Коэффициент передачи от ручки газа, rpm/градус
        self.k_fr_r = k_fr_r  # Коэффициент трения c дорогой
        self.k_fr_b = k_fr_b  # Коэффициент трения в подшипнике
        self.k_bh = k_bh  # Коэффициент передачи от руки тормоза, /градус
        self.m_pr = m_pr  # Масса ЭС без человека
        self.J_psi = J_psi  # Момент инерции ЭС в канале рыскания
        self.J_phi = (m_p * s_p) / 12 + 0.333 * m_r * l_r + m_r * d_cm * d_cm  # Момент инерции ЭС в канале крена
        self.k_r = k_r  # коэффициент эффективности руля
        self.k_gyr = k_gyr  # коэффициент гироскопического момента

        self.observation_shape = (6,)
        self.action_space = (3,)

        self.min_action = -1
        self.max_action = 1

        self.psi_pr = 0  # 1 градус в канале рыскания
        self.phi_pr = 0  # 0 градусов в канале крена
        self.velocity_pr = 10  # 2 м/с

        self.reward_best = 0

        self.y0, self.t0 = [0, 0, 0, np.radians(1), 0, 0], 0  # начальные условия
        self.done = False

    def reset(self):
        self.y0, self.t0 = [0, 0, 0, np.radians(1), 0, 0], 0  # начальные условия
        self.done = False
        return list(np.zeros(self.observation_shape[0]))


    def step(self, action):

        global done
        u = action.numpy()
        u1, u2, u3 = u
        if u2 < 0:
            u2 = 0
        # print(u1, u2, u3)

        ts = []
        ys = []

        omega_t = []
        psi_t = []
        speed_psi_t = []
        phi_t = []
        speed_phi_t = []
        velocity_t = []

        omega, psi, speed_psi, phi, speed_phi, velocity = 0, 0, 0, 0, 0, 0

        def fout(t, y):  # обработчик шага
            ts.append(t)
            ys.append(list(y.copy()))
            y1, y2, y3, y4, y5, y6 = y

            omega_t.append(y1)
            psi_t.append(y2)
            speed_psi_t.append(y3)
            phi_t.append(y4)
            speed_phi_t.append(y5)
            velocity_t.append(y6)


        # функция правых частей системы ОДУ
        def f(t, y):
            g = 9.81
            y1, y2, y3, y4, y5, y6 = y
            M = (self.k_m * self.P_m * self.k_ts * np.sin(np.radians(30*u1))) - self.k_fr_b * y1 - np.sign(y1) * (
                        self.k_fr_r + self.k_bh * np.sin(np.radians(2*u2))) * np.cos(y4) * self.m_mv * self.r_mv
            return [M / self.J_mv,
                    y3,
                    M * self.k_r * np.sin(np.radians(30*u3)) / self.J_psi,
                    y5,
                    (self.m_pr * g * np.sin(y4) - y1 * y5 * self.k_gyr) / self.J_phi,
                    -M * np.cos(y4) / (self.m_pr * self.r_mv)]

        tmax = 0.05  # максимально допустимый момент времени

        ODE = ode(f)
        ODE.set_integrator('dopri5', max_step=0.01)
        ODE.set_solout(fout)
        ODE.set_initial_value(self.y0, self.t0)  # задание начальных значений
        ODE.integrate(tmax)  # решение ОДУ

        reward = - 1 * (np.square(self.psi_pr - np.degrees(psi_t[-1]))
                         + np.square(self.phi_pr - np.degrees(phi_t[-1]))
                         + np.square(self.velocity_pr - velocity_t[-1]))

        if self.reward_best < reward:
            self.reward_best = reward

        observation_ = [omega_t[-1], np.degrees(psi_t[-1]), np.degrees(speed_psi_t[-1]), np.degrees(phi_t[-1]),
                        np.degrees(speed_phi_t[-1]), velocity_t[-1]]

        self.t0 = self.t0 + ts[-1]
        # print(self.t0, ' c. ', np.degrees(psi_t[-1]), ' град. ', np.degrees(phi_t[-1]), ' град ', velocity_t[-1],
        #       ' м/с ')

        if np.abs(np.degrees(psi_t[-1])) > 10 or np.abs(np.degrees(phi_t[-1])) > 10:
            self.done = True

        if ts[-1] > 50:
            sys.exit()


        figure_file_phi: str = 'plots/phi.png'
        plot_control(ts, np.degrees(phi_t[::-1]), figure_file_phi)

        figure_file_psi: str = 'plots/psi.png'
        plot_control(ts, np.degrees(psi_t[::-1]), figure_file_psi)

        figure_file_vel: str = 'plots/velocity.png'
        plot_control(ts, velocity_t[::-1], figure_file_vel)

        return observation_, reward, self.done

    def reward_range(self):
        return self.reward_best
