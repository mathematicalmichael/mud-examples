import dolfin as fin
# from fenicstools.Probe import Probes
from pathlib import Path
import pickle
from mud_examples.models import generate_spatial_measurements as generate_sensors_pde
from mud_examples.datasets import load_poisson
import matplotlib.pyplot as plt  # move when migrating plotting code (maybe)
import numpy as np  # only needed for band_qoi
from mud.funs import wme, mud_problem # needed for the poisson class

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


def make_reproducible_without_fenics(example, lam_true=3, input_dim=2, num_samples=None, num_measure=100):
    
    model_list = pickle.load(open(f'res{input_dim}u.pkl', 'rb'))
    if num_samples is None:
        num_samples = len(model_list)
        
    if 'alt' in example:  # alternative measurement locations for more sensitivity / precision
        sensors = generate_sensors_pde(num_measure, ymax=0.95, xmax=0.25)
    else:
        sensors = generate_sensors_pde(num_measure, ymax=0.95, xmax=0.95)

    lam, qoi = load_poisson(sensors, model_list[0:num_samples], nx=36, ny=36)
    qoi_ref = poisson_sensor_model(sensors, gamma=lam_true, nx=36, ny=36)

    pn = poissonModel(gamma=lam_true)
    c = pn.function_space().mesh().coordinates()
    v = [pn(c[i,0], c[i,1]) for i in range(len(c))]

    g = gamma_boundary_condition(lam_true)
    g_mesh = np.linspace(0, 1, 1000)
    g_plot = [g(0,y) for y in g_mesh]
    ref = {'sensors': sensors, 'lam': lam, 'qoi': qoi, 'truth': lam_true, 'data': qoi_ref,
           'plot_u': (c,v), 'plot_g': (g_mesh, g_plot) }

    fname = f'pde_{input_dim}D/{example}_ref.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(ref, f)
    print(fname, 'saved:', Path(fname).stat().st_size // 1000, 'KB')

    return fname


def plot_without_fenics(fname, num_sensors=None, num_qoi=2, mode='hor', fsize = 36, prefix=None):
    plt.figure(figsize=(10,10))
    mode = mode.lower()
    colors = ['xkcd:red', 'xkcd:black', 'xkcd:orange', 'xkcd:blue', 'xkcd:green']
    ref = pickle.load(open(fname, 'rb'))
    sensors = ref['sensors']
    qoi_ref = ref['data']
    coords, vals = ref['plot_u']
    x, y = sensors[:,0], sensors[:,1]
#     try:
#         import fenics as fin
#         from poisson import poissonModel
#         pn = poissonModel()
#         fin.plot(pn, vmin=-0.5, vmax=0)
#     except:
    plt.tricontourf(coords[:, 0], coords[:, 1], vals, levels=20, vmin=-0.5, vmax=0)

    input_dim = ref['lam'].shape[1]
    
    plt.title(f"Response Surface")
    if num_sensors is not None:  # plot sensors
        intervals = np.linspace(0, 1, num_qoi+2)[1:-1]
        if mode == 'hor':
            qoi_indices = band_qoi(sensors, num_qoi, axis=1)
            # partitions equidistant between sensors
            _intervals = np.array(intervals[1:]) + \
                           ( np.array(intervals[:-1]) - \
                           np.array(intervals[1:]) ) / 2

        elif mode == 'ver':
            qoi_indices = band_qoi(sensors, num_qoi, axis=0)
            # partitions equidistant on x_1 = (0, 1)
            _intervals = np.linspace(0, 1, num_qoi+1)[1:]

        for i in range(0, num_qoi):
            _q = qoi_indices[i][qoi_indices[i] < num_sensors ]
            plt.scatter(sensors[_q,0], sensors[_q,1], s=100, color=colors[i%2])
            if i < num_qoi - 1:
                if mode == 'hor':
                    plt.axhline(_intervals[i], lw=3, c='k')
                elif mode == 'ver':
                    plt.axvline(_intervals[i], lw=3, c='k')

        plt.scatter([0]*num_qoi, intervals, s=500, marker='^', c='w')
    plt.xlim(0,1)
    plt.ylim(0,1)
#     plt.xticks([])
#     plt.yticks([])
    plt.xlabel("$x_1$", fontsize=fsize)
    plt.ylabel("$x_2$", fontsize=fsize)

    if prefix:
        _fname = f"{prefix}_sensors_{mode}_{input_dim}D.png"
        plt.savefig(_fname, bbox_inches='tight')


# from scipy.stats import gaussian_kde as gkde
# from scipy.stats import distributions as dist

# def ratio_dci_sing(qoi):
#     kde = gkde(qoi.T)
#     ratio_eval = dist.norm.pdf(qoi)/kde.pdf(qoi.T).ravel()
#     return ratio_eval


# def ratio_dci_mult(qois):
#     nq = np.array(qois)
#     kde = gkde(nq)
#     obs = dist.norm.pdf(nq)
#     obs_eval = np.product(obs, axis=0)
#     pre_eval = kde.pdf(nq)
#     ratio_eval = np.divide(obs_eval, pre_eval)
#     return ratio_eval


def make_mud_wrapper(domain, lam, qoi, qoi_true, indices=None):
    """
    Anonymous function
    """
    def mud_wrapper(num_obs, sd):
        return mud_problem(domain=domain, lam=lam, qoi=qoi, qoi_true=qoi_true, sd=sd, num_obs=num_obs, split=indices)

    return mud_wrapper


# probably move to helpers or utils
def band_qoi(sensors, num_qoi=1, axis=1):
    intervals = np.linspace(0, 1, num_qoi+2)[1:-1]
    if axis == 1:
        _intervals = np.array(intervals[1:]) + ( np.array(intervals[:-1]) - np.array(intervals[1:]) ) / 2
    elif axis == 0:
        _intervals = np.linspace(0, 1, num_qoi+1)[1:]
    else:
        raise ValueError("axis must be 0 or 1 since the example is in 2D")
    _intervals = [0] + list(_intervals) + [1]
    qoi_indices = [np.where(np.logical_and(sensors[:, axis] > _intervals[i],
                                           sensors[:, axis] < _intervals[i+1]))[0] for i in range(num_qoi) ]
    return qoi_indices


class pdeProblem(object):
    def __init__(self, fname=None):
        self.fname = fname
        self._lam = None
        self._lam_ref = None
        self._qoi = None
        self._qoi_ref = None
        self._sensors = None
        self._domain = None
        self._u = None
        self._g = None

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, lam):
        self._lam = lam

    @property
    def lam_ref(self):
        return self._lam_ref

    @lam_ref.setter
    def lam_ref(self, lam_ref):
        if self.domain is None:
            raise AttributeError("domain not yet set.")
        min_val, max_val = 0, 4  # problem-specific
        if (lam_ref < min_val) or (lam_ref > max_val):
            raise ValueError("lam_ref must be inside domain.")
        self._lam_ref = lam_ref

    @property
    def qoi(self):
        return self._qoi

    @qoi.setter
    def qoi(self, qoi):
        self._qoi = qoi

    @property
    def qoi_ref(self):
        return self._qoi_ref

    @qoi_ref.setter
    def qoi_ref(self, qoi_ref):
        self._qoi_ref = qoi_ref

    @property
    def sensors(self):
        return self._sensors

    @sensors.setter
    def sensors(self, sensors):
        self._sensors = sensors

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = domain

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, g):
        self._g = g

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u):
        self._u = u

    def load(self, fname=None):
        if fname: 
            self.fname = fname
        else:
            fname = self.fname
        domain, sensors, lam, qoi, qoi_ref, lam_ref, u, g = load_poisson_from_disk(fname)
        self.domain = domain
        self.sensors = sensors
        self.lam = lam
        self.lam_ref = lam_ref
        self.qoi = qoi
        self.qoi_ref = qoi_ref
        self.u = u
        self.g = g

    def mud_scalar(self):
        return make_mud_wrapper(self.domain, self.lam, self.qoi, self.qoi_ref)

    def mud_vector_horizontal(self, num_qoi=None):
        if num_qoi is None:  # set output dimension = input dimension
            num_qoi = self.lam.shape[1]
        indices = band_qoi(self.sensors, num_qoi=num_qoi, axis=1)
        return make_mud_wrapper(self.domain, self.lam, self.qoi, self.qoi_ref, indices)

    def mud_vector_vertical(self, num_qoi=None):
        if num_qoi is None:  # set output dimension = input dimension
            num_qoi = self.lam.shape[1]
        indices = band_qoi(self.sensors, num_qoi=num_qoi, axis=0)
        return make_mud_wrapper(self.domain, self.lam, self.qoi, self.qoi_ref, indices)

    def plot_initial(self, save=True, **kwargs):
        self.plot(save=save,**kwargs)
        
    def plot_solutions(self, sols, num, save=True, **kwargs):
        self.plot(sols=sols, num_measurements=num, save=save, **kwargs)

    def plot(self, sols=None, num_measurements=20, example='mud', fsize=36, ftype='png', save=False):
        lam, qoi, qoi_ref, g = self.lam, self.qoi, self.qoi_ref, self.g, 
        closest_fit_index_out = np.argmin(np.linalg.norm(qoi - np.array(qoi_ref), axis=1))
        g_projected = list(lam[closest_fit_index_out, :])
        plt.figure(figsize=(10,10))

        g_mesh, g_plot = g
        intervals = list(np.linspace(0, 1, lam.shape[1]+2)[1:-1])
        # fin.plot(u_plot, mesh=mesh, lw=5, c='k', label="$g$")
        plt.plot(g_mesh, g_plot, lw=5, c='k', label="$g$")
        plt.plot([0]+intervals+[1], [0]+g_projected+[0], lw=5, c='green', alpha=0.6, ls='--', label='$\hat{g}$', zorder=5)


        if sols is not None:
            if sols.get(num_measurements, None) is None:
                raise AttributeError(f"Solutions `sols` missing requested N={num_measurements}.")
            else:
                prefix = f'pde_{lam.shape[1]}D/{example}_solutions_N{num_measurements}'
                plot_lam = np.array(sols[num_measurements])
                if 'alt' in example:
                    qmap = '$Q_{%dD}^\prime$'%lam.shape[1]
                else:
                    qmap = '$Q_{%dD}$'%lam.shape[1]
                plt.title(f'MUD Estimates for {qmap}, N={num_measurements}', fontsize=1.25*fsize)
        else:  # initial plot, first 100
            prefix = f'pde_{lam.shape[1]}D/initial'
            plot_lam = lam[0:100, :]
            plt.title('Samples from Initial Density', fontsize=1.25*fsize)

        for _lam in plot_lam:
            plt.plot([0]+intervals+[1], [0]+list(_lam)+[0], lw=1, c='purple', alpha=0.2)


        plt.xlabel("$x_2$", fontsize=fsize)
        plt.ylabel("$g(x, \lambda)$", fontsize=fsize)



        # label min(g)
        # plt.axvline(2/7, alpha=0.4, ls=':')
        # plt.axhline(-lam_true, alpha=0.4, ls=':')
        plt.ylim(-4,0)
        plt.xlim(0,1)
        plt.legend()
        if save:
            _fname = f"{prefix}.{ftype}"
            plt.savefig(_fname, bbox_inches='tight')
            print(f"Saved {_fname}")
    #     plt.show()

def load_poisson_from_disk(fname):
    try:
        ref = pickle.load(open(fname, 'rb'))
    except FileNotFoundError as e:
        raise FileNotFoundError("File missing. Run `make_reproducible_without_fenics` first.")
    lam = ref['lam']
    input_dim = lam.shape[1]
    domain = np.array([[-4,0]]*input_dim)
    qoi = ref['qoi']
    qoi_ref = ref['data']
    lam_ref = ref['truth']
    u = ref['plot_u']
    g = ref['plot_g']
    sensors = ref['sensors']
    return domain, sensors, lam, qoi, qoi_ref, lam_ref, u, g





if __name__=='__main__':

    # should we maybe change this to generate raw QoI data instead?
    u = poissonModel(gamma=3)
    fin.plot(u, interactive=True)
    # Save solution in VTK format
    file = fin.File("poisson.pvd")
    file << u
