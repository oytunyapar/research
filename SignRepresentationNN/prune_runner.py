from SignRepresentationNN.prune import *


class PruneRunnerConfiguration:
    loss_function = LossFunction.MSE
    regularization_function = RegularizationFunction.HOYER_SQUARE
    regularization_strength = 0.05
    simple_model = False


def prune_runner_single(function, dimension, pruner=None, prune_runner_configuration=None):
    if pruner is None:
        if prune_runner_configuration is None:
            prune_runner_configuration = PruneRunnerConfiguration()
        pruner = PruneSigmaPiModel(function, dimension,
                                   prune_runner_configuration.regularization_strength,
                                   prune_runner_configuration.simple_model,
                                   prune_runner_configuration.loss_function,
                                   prune_runner_configuration.regularization_function)

    if pruner.operation():
        return pruner.num_zeroed_weights()
    else:
        return -1


def prune_runner_parameters(prune_runner_configuration=None):
    if prune_runner_configuration is None:
        prune_runner_configuration = PruneRunnerConfiguration()
    return {"simple_model": str(prune_runner_configuration.simple_model),
            "reg_strength": str(prune_runner_configuration.regularization_strength),
            "loss_function": str(prune_runner_configuration.loss_function),
            "regularization_func": str(prune_runner_configuration.regularization_function)}
