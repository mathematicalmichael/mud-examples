import numpy as np


def load_poisson(sensors, file_list, nx=36, ny=36):
    from fenics import FunctionSpace, RectangleMesh, Point, Function

    num_samples = len(file_list)
    print(f"Loaded {num_samples} evaluations of parameter space.")

    mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)
    V = FunctionSpace(mesh, 'Lagrange', 1)

    qoi = []
    lam = []
    # go through all the files and load them into an array
    for i in range(num_samples):
        fname = file_list[i][i]['u']
        u = Function(V, fname)
        q = [u(xi, yi) for xi, yi in sensors]  # sensors
        qoi.append(np.array(q))
        lam.append(file_list[i][i]['gamma'])  # TODO: change name of this
    qoi = np.array(qoi)
    lam = np.array(lam)
    print('QoI', qoi.shape,
          'Input', lam.shape,
          'Measurements', sensors.shape,
          'Num Input Samples', num_samples)  # check shapes correct

    return lam, qoi
