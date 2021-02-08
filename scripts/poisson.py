import dolfin as fin
# from fenicstools.Probe import Probes
from pathlib import Path
import pickle
from mud_examples.models import generate_spatial_measurements as generate_sensors_pde
from mud_examples.datasets import load_poisson

def poissonModel(gamma=3,
                 mesh=None, width=1,
                 nx=36, ny=36):
    """
    `gamma` is scaling parameter for left boundary condition
    `n_x` and `n_y` are the number of elements for the horizontal/vertical axes of the mesh
    """
    # Create mesh and define function space
    if mesh is None:
        mesh = fin.RectangleMesh(fin.Point(0,0), fin.Point(width, 1), nx, ny)

    V = fin.FunctionSpace(mesh, "Lagrange", 1)
    u = fin.TrialFunction(V)
    v = fin.TestFunction(V)

    # Define the left boundary condition, parameterized by gamma
    left_bc = gamma_boundary_condition(gamma)

    # Define the rest of the boundary condition
    dirichlet_bc = fin.Constant(0.0) # top and bottom
    right_bc = fin.Constant(0)

    # Define integrand
    boundary_markers = get_boundary_markers_for_rect(mesh, width)
    ds = fin.Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    # Define variational problem
    f = fin.Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    a = fin.inner(fin.grad(u), fin.grad(v)) * fin.dx
    L = f * v * fin.dx + right_bc * v * ds(1) + left_bc * v * ds(3) # sum(integrals_N)

    # Define Dirichlet boundary (x = 0 or x = 1)
    def top_bottom_boundary(x):
        return x[1] < fin.DOLFIN_EPS or x[1] > 1.0 - fin.DOLFIN_EPS

    bc = fin.DirichletBC(V, dirichlet_bc, top_bottom_boundary) # this is required for correct solutions

    # Compute solution
    u = fin.Function(V)
    fin.solve(a == L, u, bc)
    return u


def gamma_boundary_condition(gamma=3):
    """
    Cannot get this to instantiate successfully in another script
    """
    if isinstance(gamma, int) or isinstance(gamma, float): # 1-D case
        # the function below will have a min at (2/7, -gamma) by design (scaling factor chosen via calculus)
        lam=gamma*823543/12500
        expr = fin.Expression(f"pow(x[1], 2) * pow(x[1] - 1, 5) * {lam}", degree=3)
    else: # Higher-D case
        expr = fin.Expression(piecewise_eval_from_vector(gamma, d=1), degree=1)
    return expr


def poisson_sensor_model(sensors, gamma, nx, ny, mesh=None):
    """
    Convenience function wrapper to just return a qoi given a parameter.
    """
    assert sensors.shape[1] == 2, "pass with shape (num_sensors, 2)"
    u = poissonModel(gamma=gamma, mesh=mesh, nx=nx, ny=ny)
    return [u(xi,yi) for xi,yi in sensors]


def evaluate_and_save_poisson(sample, save_prefix):
    """
    sample is a tuple (index, gamma)
    """
    prefix = save_prefix.replace('.pkl','')
    g = sample[1]

    # define fixed mesh to avoid re-instantiation on each call to model (how it handles mesh=None)
    nx, ny = 36, 36
    mesh = fin.RectangleMesh(fin.Point(0,0), fin.Point(1,1), nx, ny)
    u = poissonModel(gamma=g, mesh=mesh, nx=nx, ny=ny)

    # Save solution as XML mesh
    fname = f"{prefix}-data/poisson-{int(sample[0]):06d}.xml"
    fin.File(fname, 'w') << u
    # tells you where to find the saved file and what the input was to generate it.
    return {int(sample[0]): {'u': fname, 'gamma': sample[1]}}


def eval_boundary_piecewise(u, n, d=1):
    """
    Takes an Expression `u` (on unit domain) and returns the string
    for another expression based on evaluating a piecewise-linear approximation.
    The mesh is equispaced into n intervals.
    """
    dx=1/(n+1)
    intervals = [i*dx for i in range(n+2)]
    node_values = [u(0,i) for i in intervals]
    return piecewise_eval(intervals, node_values, d)

       
def piecewise_eval_from_vector(u, d=1):
    """
    Takes an iterable `u` with y-values (on interior of equispaced unit domain)
    and returns the string for an expression
    based on evaluating a piecewise-linear approximation through these points.
    """
    n = len(u)
    dx = 1/(n+1)
    intervals = [i*dx for i in range(n+2)]
    node_values = [0] + list(u) + [1]
    return piecewise_eval(intervals, node_values, d)


def piecewise_eval(xvals, yvals, d=1):
    s = ''
    for i in range(1,len(xvals)):
        start = xvals[i-1]
        end = xvals[i]
        diff = start-end
        s += f' ((x[{d}] >= {start}) && (x[{d}] < {end}))*'
        s += f'({yvals[i-1]}*((x[{d}]-{end})/{diff}) + (1 - ((x[{d}]-{end})/{diff}))*{yvals[i]} ) +'
    return s[1:-1]


def eval_boundary(u, n):
    dx=1/(n+1)
    invals = [i*dx for i in range(n+2)]
    outvals = [u(0,i) for i in invals][1:-1]
    return invals[1:-1], outvals


def expressionNorm(u,v,n=100):
    u = eval_boundary(u, n)[1] 
    v = eval_boundary(v, n)[1]
    return np.linalg.norm(np.array(u) - np.array(v))/n


def copy_expression(expression):
    u = expression
    return fin.Expression(u._cppcode, **u._user_parameters,
                          degree=u.ufl_element().degree())


def get_boundary_markers_for_rect(mesh, width=1):
    
    class BoundaryX0(fin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fin.near(x[0], 0, 1E-14)

    class BoundaryX1(fin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fin.near(x[0], width, 1E-14)

    class BoundaryY0(fin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fin.near(x[1], 0, 1E-14)

    class BoundaryY1(fin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fin.near(x[1], 1, 1E-14)

    # not sure what the first argument here does.
    boundary_markers = fin.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

    # starting from top of square, going clockwise
    # we have to instantiate a class for each boundary portion in order to mark them.
    # each operation changes the state of `boundary_markers`
    BoundaryY1().mark(boundary_markers, 0)
    BoundaryX1().mark(boundary_markers, 1)
    BoundaryY0().mark(boundary_markers, 2)
    BoundaryX0().mark(boundary_markers, 3)
    
    return boundary_markers


def make_reproducible_without_fenics(prefix, lam_true=3, input_dim=2, num_samples=None, num_measure=100, out=False):
    
    model_list = pickle.load(open(f'res{input_dim}u.pkl', 'rb'))
    if num_samples is None:
        num_samples = len(model_list)
    sensors = generate_sensors_pde(num_measure)
    lam, qoi = load_poisson(sensors, model_list[0:num_samples], nx=36, ny=36)
    qoi_ref = poisson_sensor_model(sensors, gamma=lam_true, nx=36, ny=36)

    pn = poissonModel(gamma=lam_true)
    c = pn.function_space().mesh().coordinates()
    v = [pn(c[i,0], c[i,1]) for i in range(len(c))]

    ref = {'sensors': sensors, 'lam': lam, 'qoi': qoi, 'truth': lam_true, 'data': qoi_ref, 'plot': (c,v)}

    fname = f'{prefix}_ref.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(ref, f)
    print(fname, 'saved:', Path(fname).stat().st_size // 1000, 'KB')

    if out:  # optional return
        return ref


if __name__=='__main__':

    # should we maybe change this to generate raw QoI data instead?
    u = poissonModel(gamma=3)
    fin.plot(u, interactive=True)
    # Save solution in VTK format
    file = fin.File("poisson.pvd")
    file << u
