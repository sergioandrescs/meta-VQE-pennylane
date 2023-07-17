# meta-VQE-pennylane

Notebook that run meta-VQE, opt-meta-VQE, and VQE. Code inspired on work about [The Meta-Variational Quantum Eigensolver](https://arxiv.org/pdf/2009.13545.pdf).

To run the calculationsin Pennylane [Catalyst](https://docs.pennylane.ai/projects/catalyst/en/latest/index.html) is used to acceslerate the calculations. The speed improvement is impresive, compared to the use of simple Pennylane devices.

## Ansatz

* [Simplified Two Design](https://docs.pennylane.ai/en/stable/code/api/pennylane.SimplifiedTwoDesign.html)
* [Single and Double Excitations](https://docs.pennylane.ai/en/stable/code/api/pennylane.AllSinglesDoubles.html)
* [k-UpCCGSD](https://docs.pennylane.ai/en/stable/code/api/pennylane.kUpCCGSD.html)

## Gradient Method

* Finite Differences
* Parameter-Shift Rule

## Optimizers

* Gradient Descend
* SPSA
* ADAM


