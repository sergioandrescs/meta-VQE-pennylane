from custom_optimizers import gradient_descent, spsa_optimizer, adam
import numpy as np
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from scipy.optimize import minimize

import json

print(f"Pennylane {qml.__version__}")

np.random.seed(32)

# To dump data in json file


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray) or isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# %%
# Layers and wires for circuits

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

# Building the molecular hamiltonian for H2
hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    method="pyscf",
)
RUNS = 5

SHOTS = 10000  # 1000, None. If none is used, the result is analytic

WIRES = qubits

USE_SINGLES = True
USE_DOUBLES = True

init_state = qml.qchem.hf_state(2, qubits)

# https://docs.pennylane.ai/projects/catalyst/en/latest/code/api/catalyst.grad.html -

TRAINERS = {"gradient_descent":
            {"fun": gradient_descent, "options": {
                'maxiter': 150, 'tol': 1e-4, "verbose": False}},
            "spsa": {"fun": spsa_optimizer, "options": {'maxiter': 150, 'tol': 1e-4, "verbose": False}},
            "adam": {"fun": adam, "options": {'maxiter': 150, 'tol': 1e-4, 'demon': False, "verbose": False}},
            "demon_adam": {"fun": adam, "options": {'maxiter': 150, 'tol': 1e-4, 'demon': True, "verbose": False}}}


LAYERS = 1
GRADIENT_METHOD = "ps"  # fd, ps


gradient_methods = {"fd": "finite-diff",
                    "ps": "parameter-shift", "adj": "adjoint"}


# Encoder:
# Linear -> 1
# Gaussian -> 2

ENCODER_MULTIPLIER_DICT = {1: 2,
                           2: 4}

ENCODER = 2

shared_dev = qml.device("lightning.qubit", wires=WIRES, shots=SHOTS)

train_points_indexes = [2, 6, 10, 14, 18]

test_points = np.arange(0.6, 5, 0.1)

train_points = np.array([test_points[i] for i in train_points_indexes])

test_hamiltonians = []


for d in test_points:
    coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, d])

    hamiltonian, _ = qml.qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        method="pyscf",
    )

    test_hamiltonians.append(hamiltonian)

train_hamiltonians = [test_hamiltonians[i] for i in train_points_indexes]

# %%
singles, doubles = qml.qchem.excitations(2, WIRES, 0)


def linear_encoding(param_array, r):
    """1-D array with alphas and betas. len(param_array) = 2 * len(weights) 

    Args:
        param_array (float): alphas and betas for lineasr encoding
        r(float): Hamiltonian parameter (in this case, distance)
    """
    return param_array[::2]*r + param_array[1::2]


def gaussian_encoding(param_array, r):
    """1-D array with alphas, betas, gammas and deltas. len(param_array) = 4 * len(weights) 

    Args:
        param_array (float): , betas, gammas and deltas for gaussian encoding
        r(float): Hamiltonian parameter (in this case, distance)
    """

    exp_arg = param_array[1::4]*(param_array[2::4] - r)

    return param_array[::4]*np.exp(exp_arg) + param_array[3::4]

# %%
# Get shapes


# For linear is 2
# For Gaussian is 4
ENCODING_MULTIPLIER = ENCODER_MULTIPLIER_DICT[ENCODER]

if (ENCODER == 1):
    ENCODER_FUNC = linear_encoding
else:
    ENCODER_FUNC = gaussian_encoding

# Get shapes

shape = qml.kUpCCGSD.shape(k=LAYERS, n_wires=WIRES, delta_sz=0)

num_params = (LAYERS*shape[1])


num_params_encoding = num_params*ENCODING_MULTIPLIER

weights = np.random.random(num_params_encoding)


def core_circuit(params):

    shape = qml.kUpCCGSD.shape(k=LAYERS, n_wires=WIRES, delta_sz=0)

    weights = np.reshape(params, shape)

    qml.kUpCCGSD(weights=weights, wires=range(
        WIRES), k=LAYERS, init_state=init_state)


@qml.qnode(shared_dev, diff_method=gradient_methods[GRADIENT_METHOD], shots=SHOTS)
def normal_circuit(params, hamiltonian):

    core_circuit(params)

    return qml.expval(
        qml.Hamiltonian(np.array(hamiltonian.coeffs), hamiltonian.ops)
    )


@qml.qnode(shared_dev, diff_method=gradient_methods[GRADIENT_METHOD], shots=SHOTS)
def meta_circuit(params, hamiltonian, r):

    weights = ENCODER_FUNC(params, r)

    core_circuit(weights)

    return qml.expval(
        qml.Hamiltonian(np.array(hamiltonian.coeffs), hamiltonian.ops)
    )


if (GRADIENT_METHOD == 'fd'):
    normal_grad_fun = normal_circuit.gradient_fn(
        normal_circuit, h=1e-3, shots=SHOTS)
    meta_grad_fun = meta_circuit.gradient_fn(meta_circuit, h=1e-3, shots=SHOTS)
else:
    normal_grad_fun = normal_circuit.gradient_fn(normal_circuit, shots=SHOTS)
    meta_grad_fun = meta_circuit.gradient_fn(meta_circuit, shots=SHOTS)


def meta_cost(params, train_hamiltonians, return_grad=True):

    energies = []
    gradients = []

    for count, train_ham in enumerate(train_hamiltonians):

        distance = train_points[count]
        distance.requires_grad = False

        energies.append(meta_circuit(params, train_ham, distance))
        if (return_grad):
            gradients.append(meta_grad_fun(
                params, train_ham, distance))

    join_energy = np.sum(np.array(energies))

    if (return_grad):
        all_gradients = np.array(gradients)
        joined_gradients = np.sum(
            np.array([all_gradients[i][0] for i in range(len(all_gradients))]), axis=0)

        return join_energy/len(train_points), joined_gradients/len(train_points)
    else:
        return join_energy/len(train_points)

# %%


def normal_cost(params, hamiltonian, return_grad=True):

    if (return_grad):
        return normal_circuit(params, hamiltonian), np.array(normal_grad_fun(params, hamiltonian))
    else:
        return normal_circuit(params, hamiltonian)


# %%
for num_run in range(RUNS):

    TRAINING_ENERGIES = {"gradient_descent": {
        "meta": {"energies": [], "runs": 0, "trained_vars": [], "energies_after_meta": []},
        "opt_meta": {"energies": [], "runs_per_step": [], "trained_vars": []},
        "vqe": {"energies": [], "runs_per_step": [], "trained_vars": []}
    },
        "spsa": {
        "meta": {"energies": [], "runs": 0, "trained_vars": [], "energies_after_meta": []},
        "opt_meta": {"energies": [], "runs_per_step": [], "trained_vars": []},
        "vqe": {"energies": [], "runs_per_step": [], "trained_vars": []}
    },
        "adam": {
        "meta": {"energies": [], "runs": 0, "trained_vars": [], "energies_after_meta": []},
        "opt_meta": {"energies": [], "runs_per_step": [], "trained_vars": []},
        "vqe": {"energies": [], "runs_per_step": [], "trained_vars": []}
    },
        "demon_adam": {
        "meta": {"energies": [], "runs": 0, "trained_vars": [], "energies_after_meta": []},
        "opt_meta": {"energies": [], "runs_per_step": [], "trained_vars": []},
        "vqe": {"energies": [], "runs_per_step": [], "trained_vars": []}
    },

    }

    for train_method in TRAINERS:

        params_run = np.random.random(num_params_encoding)

        options = TRAINERS[train_method]["options"]
        options.update({"hamiltonian": train_hamiltonians})

        res_train = minimize(meta_cost, params_run,
                             method=TRAINERS[train_method]["fun"], options=options)

        TRAINING_ENERGIES[train_method]["meta"]["trained_vars"] = res_train.x
        TRAINING_ENERGIES[train_method]["meta"]["energies"].append(
            res_train.fun)
        TRAINING_ENERGIES[train_method]["meta"]["runs"] = res_train.nit

    # %%
    for count, d in enumerate(test_points):
        for train_method in TRAINERS:
            encoded_params = ENCODER_FUNC(
                TRAINING_ENERGIES[train_method]["meta"]["trained_vars"], d)
            TRAINING_ENERGIES[train_method]["meta"]["energies_after_meta"].append(
                normal_cost(encoded_params, test_hamiltonians[count], return_grad=False))

    # %%
    for count, work_hamiltonian in enumerate(test_hamiltonians):

        init_params_run = ENCODER_FUNC(np.array(
            TRAINING_ENERGIES[train_method]["meta"]["trained_vars"]), test_points[count])

        for train_method in TRAINERS:

            options = TRAINERS[train_method]["options"]
            options.update({"hamiltonian": work_hamiltonian})

            res_train = minimize(normal_cost, init_params_run,
                                 method=TRAINERS[train_method]["fun"], options=options)

            TRAINING_ENERGIES[train_method]["opt_meta"]["trained_vars"].append(
                res_train.x)
            TRAINING_ENERGIES[train_method]["opt_meta"]["energies"].append(
                res_train.fun)
            TRAINING_ENERGIES[train_method]["opt_meta"]["runs_per_step"].append(
                res_train.nit)

            if (count == 0):
                params_run = np.random.random(num_params)
            else:
                params_run = TRAINING_ENERGIES[train_method]["vqe"]["trained_vars"][-1]

            res_train_vqe = minimize(
                normal_cost, params_run, method=TRAINERS[train_method]["fun"], options=options)

            TRAINING_ENERGIES[train_method]["vqe"]["trained_vars"].append(
                res_train_vqe.x)
            TRAINING_ENERGIES[train_method]["vqe"]["energies"].append(
                res_train_vqe.fun)
            TRAINING_ENERGIES[train_method]["vqe"]["runs_per_step"].append(
                res_train_vqe.nit)

    print(f'run {num_run} done!')

    with open(f'exp_runs/kupccgsd_{LAYERS}layers_{GRADIENT_METHOD}_{num_run}.json', 'w') as f:
        f.write(json.dumps({
            "layers": LAYERS,
            "results": TRAINING_ENERGIES
        }, indent=4, cls=NumpyEncoder))
