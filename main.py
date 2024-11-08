import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def target_function(x):
    return (x[0] ** 2 + 3 * x[1] ** 2 + 2 * x[0] * x[1])


def initialize_particles(num_particles, bounds, initial_inertia):
    global positions, velocities, pbest_positions, pbest_scores, gbest_position
    positions = np.random.uniform(bounds[0], bounds[1], (num_particles, 2))
    velocities = np.random.uniform(-1, 1, (num_particles, 2))
    pbest_positions = np.copy(positions)
    pbest_scores = np.array([target_function(p) for p in pbest_positions])
    gbest_position = pbest_positions[np.argmin(pbest_scores)]
    iteration_var.set(0)


def update_particles_standard(iteration, num_particles, bounds, c1, c2, initial_inertia):
    global positions, velocities, pbest_positions, pbest_scores, gbest_position
    inertia = initial_inertia

    for i in range(num_particles):
        velocities[i] = (
                inertia * velocities[i]
                + c1 * np.random.rand() * (pbest_positions[i] - positions[i])
                + c2 * np.random.rand() * (gbest_position - positions[i])
        )

        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])

        score = target_function(positions[i])
        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = positions[i]

    gbest_position = pbest_positions[np.argmin(pbest_scores)]
    gbest_score = pbest_scores[np.argmin(pbest_scores)]

    best_solution.set(f"X[0] = {gbest_position[0]:.4f}\nX[1] = {gbest_position[1]:.4f}")
    function_value.set(f"{gbest_score:.4f}")
    plot_particles()


def update_particles_modified(iteration, num_particles, bounds, c1, c2, initial_inertia, final_inertia, num_iterations):
    global positions, velocities, pbest_positions, pbest_scores, gbest_position
    inertia = initial_inertia - (initial_inertia - final_inertia) * (iteration / num_iterations)

    for i in range(num_particles):
        velocities[i] = (
                inertia * velocities[i]
                + c1 * np.random.rand() * (pbest_positions[i] - positions[i])
                + c2 * np.random.rand() * (gbest_position - positions[i])
        )

        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])

        score = target_function(positions[i])
        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = positions[i]

    gbest_position = pbest_positions[np.argmin(pbest_scores)]
    gbest_score = pbest_scores[np.argmin(pbest_scores)]

    best_solution.set(f"X[0] = {gbest_position[0]:.4f}\nX[1] = {gbest_position[1]:.4f}")
    function_value.set(f"{gbest_score:.4f}")
    plot_particles()


def plot_particles():
    ax.clear()
    ax.scatter(positions[:, 0], positions[:, 1], s=10, color="black", label="Particles")
    ax.scatter(gbest_position[0], gbest_position[1], color="red", s=50, label="Best Solution")
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_title("Распределение частиц")
    ax.legend()
    canvas.draw()


def run_standard_pso():
    num_particles = num_particles_var.get()
    bounds = (-500, 500)
    initial_inertia = inertia_var.get()
    c1 = c1_var.get()
    c2 = c2_var.get()
    num_iterations = iterations_var.get()

    initialize_particles(num_particles, bounds, initial_inertia)
    for i in range(num_iterations):
        update_particles_standard(i, num_particles, bounds, c1, c2, initial_inertia)
        if i % 10 == 0 or i == num_iterations - 1:
            iteration_var.set(i + 1)


def run_modified_pso():
    num_particles = num_particles_var.get()
    bounds = (-500, 500)
    initial_inertia = inertia_var.get()
    final_inertia = final_inertia_var.get()
    c1 = c1_var.get()
    c2 = c2_var.get()
    num_iterations = iterations_var.get()

    initialize_particles(num_particles, bounds, initial_inertia)
    for i in range(num_iterations):
        update_particles_modified(i, num_particles, bounds, c1, c2, initial_inertia, final_inertia, num_iterations)
        if i % 10 == 0 or i == num_iterations - 1:
            iteration_var.set(i + 1)


root = tk.Tk()
root.title("Роевой интеллект")

frame_params = ttk.LabelFrame(root, text="Параметры")
frame_params.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

tk.Label(frame_params, text="Функция:").grid(row=0, column=0, sticky="w")
tk.Label(frame_params, text="(x[1]**2 + 3 * x[2]**2 + 2 * x[1] * x[2])").grid(row=0, column=1, sticky="w")

tk.Label(frame_params, text="Начальная инерция:").grid(row=1, column=0, sticky="e")
inertia_var = tk.DoubleVar(value=0.9)
tk.Entry(frame_params, textvariable=inertia_var).grid(row=1, column=1)

tk.Label(frame_params, text="Конечная инерция:").grid(row=2, column=0, sticky="e")
final_inertia_var = tk.DoubleVar(value=0.4)
tk.Entry(frame_params, textvariable=final_inertia_var).grid(row=2, column=1)

tk.Label(frame_params, text="Коэф. собственного лучшего значения:").grid(row=3, column=0, sticky="e")
c1_var = tk.DoubleVar(value=2)
tk.Entry(frame_params, textvariable=c1_var).grid(row=3, column=1)

tk.Label(frame_params, text="Коэф. глобального лучшего значения:").grid(row=4, column=0, sticky="e")
c2_var = tk.DoubleVar(value=5)
tk.Entry(frame_params, textvariable=c2_var).grid(row=4, column=1)

tk.Label(frame_params, text="Количество частиц:").grid(row=5, column=0, sticky="e")
num_particles_var = tk.IntVar(value=300)
ttk.Spinbox(frame_params, from_=1, to=1000, textvariable=num_particles_var).grid(row=5, column=1)

frame_results = ttk.LabelFrame(root, text="Результаты")
frame_results.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

tk.Label(frame_results, text="Лучшее решение:").grid(row=0, column=0, sticky="w")
best_solution = tk.StringVar()
tk.Label(frame_results, textvariable=best_solution, justify="left").grid(row=1, column=0, sticky="w")

tk.Label(frame_results, text="Значение функции:").grid(row=2, column=0, sticky="w")
function_value = tk.StringVar()
tk.Label(frame_results, textvariable=function_value).grid(row=3, column=0, sticky="w")

frame_control = ttk.LabelFrame(root, text="Управление")
frame_control.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")

tk.Label(frame_control, text="Количество итераций:").grid(row=1, column=0, sticky="e")
iterations_var = tk.IntVar(value=100)
ttk.Spinbox(frame_control, from_=1, to=1000, textvariable=iterations_var).grid(row=1, column=1)

tk.Button(frame_control, text="Рассчитать (Обычный)", command=run_standard_pso).grid(row=2, column=0, columnspan=2)
tk.Button(frame_control, text="Рассчитать (Модифицированный)", command=run_modified_pso).grid(row=3, column=0,
                                                                                              columnspan=2)

iteration_var = tk.IntVar(value=0)
tk.Label(frame_control, text="Выполнено итераций:").grid(row=4, column=0, sticky="e")
tk.Label(frame_control, textvariable=iteration_var).grid(row=4, column=1, sticky="w")

frame_solution = ttk.LabelFrame(root, text="Решение")
frame_solution.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")

fig, ax = plt.subplots()
ax.set_xlim((-500, 500))
ax.set_ylim((-500, 500))
canvas = FigureCanvasTkAgg(fig, master=frame_solution)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

root.mainloop()
