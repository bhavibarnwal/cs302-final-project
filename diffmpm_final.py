import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 20
act_strength = 4

def allocate_fields():
    # put in 4000 as max_particles instead of n_particles
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, 4000).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, 4000).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)

    ti.root.lazy_grad()

@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                          2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


@ti.kernel
def compute_loss():
    dist = x_avg[None][0]
    loss[None] = -dist


@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def add_circle(self, x, y, r, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        
        density = 1 / (dx * dx)                     # target particle density
        n_circle = int(math.pi * r**2 * density)
        real_dx = math.sqrt((math.pi * r**2) / n_circle)
        w_count = int(r / real_dx) * 2       # w_count: # of particles along 1 axis of circle, scaled by dx = 1 / n_grid
        # print(n_circle)
        
        # real_dx = 2 * r / w_count       # adjusted spacing to fit particles within entire diameter (but it's just equal to dx then?)
        # print(f"Radius: {r}, real_dx: {2*r/w_count}, w_count: {w_count}")
        for i in range(w_count):
            for j in range(w_count):
                px = x + (i + 0.5) * real_dx 
                py = y + (j + 0.5) * real_dx 
                px2 = x - (i + 0.5) * real_dx 
                py2 = y - (j + 0.5) * real_dx 

                distance1 = (px - x)**2 + (py - y)**2
                distance2 = (px2 - x)**2 + (py2 - y)**2
                distance3 = (px - x)**2 + (py2 - y)**2
                distance4 = (px2 - x)**2 + (py - y)**2
                # print(f"Particle position: ({px}, {px2}), Distance from center: {distance1, distance2}, Inside circle: {distance1 <= r**2 or distance2 <= r**2}")

                if distance1 <= r**2 and [px + self.offset_x, py + self.offset_y] not in self.x:        # maintains particles within circular boundary
                    self.x.append([px + self.offset_x, py + self.offset_y])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)
                if distance2 <= r**2 and [px2 + self.offset_x, py2 + self.offset_y] not in self.x:
                    self.x.append([px2 + self.offset_x, py2 + self.offset_y])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)
                if distance3 <= r**2 and [px, py2] not in self.x:
                    self.x.append([px + self.offset_x, py2 + self.offset_y])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)
                if distance4 <= r**2 and [px2 + self.offset_x, py + self.offset_y] not in self.x:
                    self.x.append([px2 + self.offset_x, py + self.offset_y])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)
                else:
                    pass
                    # print("false")

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act


def fish(scene):
    scene.add_rect(0.025, 0.025, 0.95, 0.1, -1, ptype=0)
    scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
    scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
    scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
    scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
    scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
    scene.set_n_actuators(4)


def robot(scene):
    scene.set_offset(0.1, 0.03)
    scene.add_rect(0.0, 0.1, 0.3, 0.1, -1)      # body
    scene.add_rect(0.0, 0.0, 0.05, 0.1, 0)      # leg 1, L half
    scene.add_rect(0.05, 0.0, 0.05, 0.1, 1)     # leg 1, R half
    scene.add_rect(0.2, 0.0, 0.05, 0.1, 2)      # leg 2, L half
    scene.add_rect(0.25, 0.0, 0.05, 0.1, 3)     # leg 2, R half
    scene.add_circle(0.3, 0.2, 0.1, 4)          # head
    scene.add_circle(0.0, 0.2, 0.04, 5)         # tail
    scene.set_n_actuators(6)


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    aid = actuator_id.to_numpy()[:n_particles]
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s, :n_particles]
    # print(f"Frame {s}: n_particles = {len(particles)}, aid.shape = {aid.shape}, colors.shape = {colors.shape}")
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            try:
                act = actuation_[s - 1, int(aid[i])]
            except:
                print(actuation_.shape)
                print(i)
                print(aid[i])
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

def build_robot_skeleton(scene, num_segments, segment_size, body_width, body_height):

    # builds caterpillar-like creature

    scene.set_offset(0.1, 0.03) 
    actuation_id = 0
    seg_spacing = body_width / (num_segments * 2)   
    for i in range(0, num_segments):
        x_seg = i * seg_spacing             # x pos for body segment
        x_joint = (i+1) * seg_spacing       # x pos for joint between segments
        x_leg = x_seg + segment_size/4      # x pos for leg attached to segment

        scene.add_rect(x_seg, body_height, segment_size, segment_size, actuation_id)
        actuation_id += 1

        if i != num_segments-1:
            scene.add_rect(x_joint - segment_size/2, body_height+(segment_size-0.01)/2, segment_size, 0.01, actuation_id)
            actuation_id += 1

        scene.add_rect(x_leg, 0, segment_size / 2, body_height, actuation_id)
        actuation_id += 1

    # if random.random() > 0.5:  # 50% chance of a head
    #     scene.add_rect(x_seg + segment_size/2, body_height+segment_size-0.01, segment_size / 2, 0.01, actuation_id)
    #     actuation_id += 1
    #     scene.add_circle(x_seg + segment_size, body_height+segment_size, segment_size / random.uniform(1,3), actuation_id)  # head is of random radius
    #     actuation_id += 1

    scene.set_n_actuators(actuation_id + 1)  

def main():
    
    best_dist, all_data = [], []
    global n_particles
    h1 = float(0.1)                             # mutating height
    seg1 = 9                                    # mutating segment number
    h2, h3, h4 = np.random.uniform(0.01, 0.1), np.random.uniform(0.01, 0.1), np.random.uniform(0.01, 0.1)    
    seg2, seg3, seg4 = np.random.choice(np.arange(1,10)), np.random.choice(np.arange(1,10)), np.random.choice(np.arange(1,10))

    for gen in range(5):

        distances = [0, 0, 0, 0]
        all_losses = [[], [], [], []]
        heights = [h1, h2, h3, h4]
        segments = [seg1, seg2, seg3, seg4]

        for type in range(4):

            parser = argparse.ArgumentParser()
            parser.add_argument('--iters', type=int, default=100)
            options = parser.parse_args()

            scene = Scene()
            build_robot_skeleton(scene, segments[type], 0.05, segments[type]*0.05*2.5, heights[type])
            scene.finalize()
            print(f"OFFSPRING {type+1}\nHEIGHT: {heights[type]:.3f}, SEGMENT NUMBER: {segments[type]}")
            
            if gen == 0 and type == 0:
                allocate_fields()

            for i in range(n_actuators):
                for j in range(n_sin_waves):
                    weights[i, j] = np.random.randn() * 0.01       

            for i in range(scene.n_particles):
                x[0, i] = ti.Vector(scene.x[i])
                F[0, i] = [[1, 0], [0, 1]]
                actuator_id[i] = int(scene.actuator_id[i])  
                particle_type[i] = scene.particle_type[i]

            losses = []

            for iter in range(options.iters):
                with ti.ad.Tape(loss):
                    forward()
                l = loss[None]
                losses.append(l)
                print('i=', iter, 'loss=', l)
                learning_rate = 0.1

                for i in range(n_actuators):
                    for j in range(n_sin_waves):
                        # print(weights.grad[i, j])
                        weights[i, j] -= learning_rate * weights.grad[i, j]
                    bias[i] -= learning_rate * bias.grad[i]

                if iter % 10 == 0:
                    # visualize
                    forward(1500)
                    for s in range(15, 1500, 16):
                        visualize(s, 'diffmpm/iter{:03d}/'.format(iter))

            distances[type] = (type+1, -l)
            all_losses[type] = losses

        plt.title(f"Generation {gen+1}: Optimization of Initial Velocity")
        plt.ylabel("Loss")
        plt.xlabel("Gradient Descent Iterations")
        plt.plot(range(options.iters), all_losses[0], label=f"Offspring 1: height = {h1}, {seg1} segments")
        plt.plot(range(options.iters), all_losses[1], label=f"Offspring 2: height = {h2}, {seg2} segments")
        plt.plot(range(options.iters), all_losses[2], label=f"Offspring 3: height = {h3}, {seg3} segments")
        plt.plot(range(options.iters), all_losses[3], label=f"Offspring 4: height = {h4}, {seg4} segments")
        plt.legend()
        plt.show()

        all_data.append([(h1, seg1, distances[0]), (h2, seg2, distances[1]), (h3, seg3, distances[2]), (h4, seg4, distances[3])])

        sorted_dist = sorted(distances, key=lambda x: x[1])
        print(sorted_dist)
        heights = [h1, h2, h3, h4]
        segments = [seg1, seg2, seg3, seg4]
        first = sorted_dist[3][0] - 1
        second = sorted_dist[2][0] - 1
        third = sorted_dist[1][0] - 1

        # store best creature of generation
        best_dist.append((gen+1, heights[first], segments[first], sorted_dist[3][1])) 
        
        # purebred offspring of both best-performing organisms 
        h1 = heights[first]     
        seg1 = segments[first]

        # offspring of 1st place + 2nd place organisms
        # print(f"averaging: {heights[first]}, {heights[second]}")
        h2 = np.mean([heights[first], heights[second]])       
        seg2 = int(round(np.mean([segments[first], segments[second]])))

        # offspring of 2nd place + 3rd place organisms
        # print(f"averaging: {heights[second]}, {heights[third]}")
        h3 = np.mean([heights[second], heights[third]])       
        seg3 = int(round(np.mean([segments[second], segments[third]])))

        # random offspring
        h4 = np.random.uniform(0.01, 0.1)                     
        seg4 = np.random.choice(np.arange(1,10))

        clear_actuation_grad()
        clear_particle_grad()
        clear_grid()

    # DATA VISUALIZATION
    generations = np.array([i[0] for i in best_dist])  # Generation numbers
    best_heights = np.array([i[1] for i in best_dist])  # Best height per generation
    best_segments = np.array([i[2] for i in best_dist])  # Best segment count per generation
    best_losses = np.array([-i[3] for i in best_dist])  # Best loss per generation
    
    h1_values, h2_values, h3_values, h4_values = [], [], [], []
    seg1_values, seg2_values, seg3_values, seg4_values = [], [], [], []
    all_heights, all_segments, all_distances = [], [], []

    for generation in all_data:
        h1_values.append(generation[0][0])  # h1
        h2_values.append(generation[1][0])  # h2
        h3_values.append(generation[2][0])  # h3
        h4_values.append(generation[3][0])  # h4

        seg1_values.append(generation[0][1])  # seg1
        seg2_values.append(generation[1][1])  # seg2
        seg3_values.append(generation[2][1])  # seg3
        seg4_values.append(generation[3][1])  # seg4

        for h, seg, dist in generation:
            all_heights.append(h)
            all_segments.append(seg)
            all_distances.append(-dist[0])  

    # Loss v. Height
    plt.scatter(all_heights, all_distances)
    plt.xlabel("Height")
    plt.ylabel("Loss")
    plt.title("Loss vs. Height")
    plt.show()

    # Loss v. Segment Number
    plt.scatter(all_segments, all_distances)
    plt.xlabel("Segment Number")
    plt.ylabel("Loss")
    plt.title("Loss vs. Segment Number")
    plt.show()

    # Height Evolution Over Generations
    plt.scatter(generations, h1_values, label="Height 1", color="red")
    plt.scatter(generations, h2_values, label="Height 2", color="orange")
    plt.scatter(generations, h3_values, label="Height 3", color="green")
    plt.scatter(generations, h4_values, label="Height 4", color="blue")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Height")
    plt.title("Height Evolution over Generations")
    plt.show()

    # Segment Number Evolution Over Generations
    plt.scatter(generations, seg1_values, label = "Seg 1", color="red", alpha=0.7)
    plt.scatter(generations, seg2_values, label = "Seg 2", color="orange", alpha=0.7)
    plt.scatter(generations, seg3_values, label = "Seg 3", color="green", alpha=0.7)
    plt.scatter(generations, seg4_values, label = "Seg 4", color="blue", alpha=0.7)
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Segment Number")
    plt.title("Segment Number Evolution over Generations")
    plt.show()

    # Best Loss Evolution Over Generations
    plt.plot(generations, best_losses)
    plt.scatter(np.argmin(best_losses)+1, min(best_losses), color='red', marker='*', s=150, label="Best Overall Loss")
    plt.xlabel("Generation")
    plt.ylabel("Best Loss")
    plt.title("Best Losses Over Generations")
    plt.show()

    print(f"Best Generation: {np.argmin(best_losses)+1}, Best Distance: {min(best_losses)}, Best Height: {best_heights[np.argmin(best_losses)]}, Best Segment Number: {best_segments[np.argmin(best_losses)]}")

if __name__ == '__main__':
    main() 