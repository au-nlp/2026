# 02 - NN basics

This tutorial builds the core components of a neural network from scratch in PyTorch (activation functions, the perceptron, the linear layer, the MLP and the MSE loss) and then puts them together to train and evaluate an image classifier on FashionMNIST.

## Notebooks
- `0. MLP and Training.ipynb` — the exercise notebook. Work through it in order and fill in every `# TODO`.
- `0. MLP and Training - Solution.ipynb` — the worked solutions. Try to finish a part before looking at them.

## What you will practice
1. **Activation functions** — ReLU and Sigmoid, and their derivatives
2. **The perceptron** — a single neuron: weighted sum + bias + activation
3. **The linear layer** — many perceptrons in one matrix multiplication
4. **The MLP** — linear layers with nonlinearities in between
5. **MSE loss** — the loss and its gradient
6. **The whole pipeline** — Datasets, DataLoaders, the training loop, evaluation and saving a model
7. **MCQs** — to check your understanding

Every implementation is followed by a verification cell that compares your code against PyTorch's built-in `torch.nn` modules, so you can check your work as you go.

## Before you start
The notebook needs `torch`, `torchvision` and `matplotlib`, all part of the updated project environment (`uv sync`). FashionMNIST (~30 MB) downloads automatically into `./data` the first time you run the notebook, so make sure you have a working internet connection. A GPU is not required.

Please give us feedback for this tutorial!

![LS Feedback 2](../QRs/qrf2.png)
