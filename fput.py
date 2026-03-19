import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from numpy import cos, sin
from scipy.fft import dst, idst

# define contsants

N = 100 # number of moving particles
K = 10.0
alpha = 1.0     # quadratic factor
beta = 0.0      # cubic factor
rest_distance = 1.0

initial_p = np.array([1 for _ in range(N)])
# initial_p = np.zeros(N)
# initial_p[0] = 5

initial_q = np.zeros(N)
displacement = np.arange(1, N+1, 1)

dt = 0.01
FPS = 60
TIME_SCALE = 10
TIME_WINDOW = 100

def spring_force(d):
    return K*d + alpha*(d**2)

def spring_energy(d):
    return 1/2*K*(d**2) + 1/3*alpha*(d**3)


def hamiltonian(t, p, q):
    T = 0.5*np.sum(p * p)
    U = sum(spring_energy(q[i] - q[i-1]) for i in range(1, N)) + spring_energy(q[0]) + spring_energy(-q[N-1])

    return T + U


def f(t, p, q):
    dhdq = np.zeros(N)
    dhdp = p

    for i in range(0, N):
        ext_left = q[i] - q[i-1] if i > 0 else q[i]
        ext_right = q[i+1] - q[i] if i < N-1 else -q[i]
        dhdq[i] = spring_force(ext_left) - spring_force(ext_right)

    return dhdq, dhdp

def euler_simplettic(t, p, q):
    dhdq, dhdp = f(t, p, q)

    next_p = p - dt * dhdq

    dhdq, dhdp = f(t, next_p, q)
    next_q = q + dt * dhdp

    return t+dt, next_p, next_q

# might be bugged
def verlet_stormer(t, p, q):
    dhdq, dhdp = f(t, p, q)

    half_p = p - 1/2 * dt * dhdq

    dhdq, dhdp = f(t, half_p, q)
    next_q = q + dt * dhdp

    dhdq, dhdp = f(t, half_p, next_q)
    next_p = half_p - 1/2 * dt * dhdq

    return t+dt, next_p, next_q


t = [0]
states_p = [initial_p]
states_q = [initial_q]
states_h = [hamiltonian(0, initial_p, initial_q)]

def numeric_iteration(numeric_method = euler_simplettic):
    new_t, new_state_p, new_state_q = numeric_method(t[-1], states_p[-1], states_q[-1])
    t.append(new_t)
    states_p.append(new_state_p)
    states_q.append(new_state_q)
    states_h.append(hamiltonian(new_t, new_state_p, new_state_q))

#plotting

fig = plt.figure(figsize=(10, 20))
ax = fig.add_subplot(2,2,1,autoscale_on=False, xlim=(0, N+1), ylim=(-N/2, N/2))
ax.set_aspect('equal')
ax.grid()

points, = ax.plot([], [], 'o')

ax2 = fig.add_subplot(2,2,2,autoscale_on=False, xlim=(0, N+1), ylim=(-N/2, N/2))
ax2.set_aspect('equal')
ax2.grid()
vertical_points, = ax2.plot([], [], 'g-')

ax3 = fig.add_subplot(2,2,(3,4),autoscale_on=False, xlim=(0, TIME_WINDOW), ylim=(9/10*states_h[0], 11/10*states_h[0]))
ax3.grid()
energy_level, = ax3.plot([], [], '-')

def animate(frame):

    curr_t = frame / FPS * TIME_SCALE
    i = int(curr_t / dt)

    while len(t) <= i:
        numeric_iteration()


    if curr_t > TIME_WINDOW:
        ax3.set_xlim((curr_t - TIME_WINDOW, curr_t + 0.01 * TIME_WINDOW))

    start_index = int(max(0, (curr_t - TIME_WINDOW) / dt))
    h_slice = states_h[start_index:i]

    if h_slice:
        min_h = min(h_slice)
        max_h = max(h_slice)

        padding = (min_h - max_h)*0.05 if min_h != max_h else 1.0

        print(min_h, max_h)

        ax3.set_ylim((min_h - padding, max_h + padding))
        
    points.set_data(displacement + states_q[i], np.zeros(N))
    vertical_points.set_data(np.concatenate(([0], displacement, [N+1])), np.concatenate(([0], states_q[i], [0])))
    energy_level.set_data(t[start_index:i], states_h[start_index:i])

    return points, vertical_points, energy_level

ani = animation.FuncAnimation (
    fig, animate, 10000, interval=1000 / FPS, blit=False
)

plt.show()
