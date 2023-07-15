from pennylane import numpy as np
import pennylane as qml

from scipy.optimize import OptimizeResult


def gradient_descent(fun, x0, stepsize=0.1, tol=1e-4, maxiter=100,  verbose=False, **options):

    new_params = np.array(x0)
    ref_energy, grad = fun(new_params)

    energy_evolution = [ref_energy]

    niter = 0

    for i in range(maxiter):
        niter += 1

        new_params -= grad*stepsize

        new_energy, grad = fun(new_params)

        energy_evolution.append(new_energy)

        if (verbose and i % 10 == 0):
            print("Gradient Descent - Step: ", i, " Cost: ", new_energy)

        if (np.abs(new_energy-ref_energy) < tol):
            break
        else:
            ref_energy = new_energy

    print("Finished Gradient Descent training")

    return OptimizeResult(x=new_params, nit=niter, fun=energy_evolution)


def spsa_optimizer(fun, x0, maxiter=100, alpha=0.602, gamma=0.101, c=0.2, A=None, a=None, tol=1e-4, verbose=False, **options):
    new_params = np.array(x0)
    ref_energy, _ = fun(new_params)

    energy_evolution = [ref_energy]

    if not A:
        A = maxiter * 0.1

    if not a:
        a = 0.05 * (A + 1) ** alpha

    niter = 0

    for i in range(maxiter):
        niter += 1

        ak = a/np.power(i+1+A, alpha)
        ck = c/np.power(i+1, gamma)

        delta = np.random.choice([-1, 1], size=x0.shape)

        thetaplus = new_params+ck*delta
        thetaminus = new_params-ck*delta
        yplus, _ = fun(thetaplus)
        yminus, _ = fun(thetaminus)

        grad = np.array([(yplus - yminus) / (2 * ck * di) for di in delta])

        new_params -= ak*grad

        new_energy, _ = fun(new_params)

        energy_evolution.append(new_energy)

        if (verbose and i % 10 == 0):
            print("SPSA - Step: ", i, " Cost: ", new_energy)

        if (np.abs(new_energy-ref_energy) < tol):
            break
        else:
            ref_energy = new_energy

    print("Finished SPSA training")

    return OptimizeResult(x=new_params, nit=niter, fun=energy_evolution)


def adam(fun, x0, maxiter=100, stepsize=0.01, beta1=0.9, beta2=0.99, tol=1e-4, eps=1e-08, verbose=False, demon=False, adamax=False, weight_decay=0, **options):
    new_params = np.array(x0)
    ref_energy, grad = fun(new_params)

    energy_evolution = [ref_energy]

    m = np.zeros_like(x0)
    v = np.zeros_like(x0)

    niter = 0

    v_hat_use = np.zeros_like(x0)

    for i in range(maxiter):
        niter += 1

        if demon:
            p_t = 1 - i / maxiter
            beta1_use = beta1 * (p_t / (1 - beta1 + beta1 * p_t))
        else:
            beta1_use = beta1

        m = beta1_use*m + (1-beta1_use)*grad
        v = beta2*v + (1-beta2)*np.square(grad)

        mhat = m/(1-beta1_use**(i+1))

        vhat = v/(1-beta2**(i+1))

        if adamax:
            v_hat_use = np.maximum(v_hat_use, np.abs(vhat))
        else:
            v_hat_use = vhat

        step = mhat / \
            (np.array([np.sqrt(vhat_i) + eps for vhat_i in v_hat_use])
             ) + new_params*weight_decay

        new_params -= stepsize*step

        new_energy, grad = fun(new_params)

        energy_evolution.append(new_energy)

        if (verbose and i % 10 == 0):
            print("ADAM - Step: ", i, " Cost: ", new_energy)

        if (np.abs(new_energy-ref_energy) < tol):
            break
        else:
            ref_energy = new_energy

    print("Finished ADAM training")

    return OptimizeResult(x=new_params, nit=niter, fun=energy_evolution)
