import torch
import numpy
from SignRepresentationNN.models import *
from SignRepresentationNN.data import *
from enum import Enum


class LossFunction(Enum):
    MSE = 0
    RELU = 1
    EXPONENTIAL = 2


class RegularizationFunction(Enum):
    HOYER_SQUARE = 0
    L1 = 1
    HOYER_SQUARE_AND_L1_AND = 2


class PruneSigmaPiModel:
    def __init__(self, function, dimension, regularization_strength=0.05, simple_model=False,
                 loss_function=LossFunction.MSE, regularization_function=RegularizationFunction.HOYER_SQUARE):
        self.dimension = dimension
        self.two_to_power_dimension = 2 ** dimension
        self.total_number_of_functions = 2 ** self.two_to_power_dimension

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.simple_model = simple_model
        if simple_model:
            self.model = SigmaPiSimpleModel(dimension).to(self.device)
        else:
            self.model = SigmaPiModel(dimension).to(self.device)

        self.loss_function_enum = loss_function

        if loss_function == LossFunction.MSE:
            self.loss_function = torch.nn.functional.mse_loss
        elif loss_function == LossFunction.RELU:
            self.loss_function = self.relu_error
        elif loss_function == LossFunction.EXPONENTIAL:
            self.loss_function = self.exponential_error
        else:
            raise Exception("Unknown loss function.")

        self.regularization_function_enum = regularization_function

        if regularization_function == RegularizationFunction.HOYER_SQUARE:
            self.regularization_function = self.hoyer_square_regularization_func
        elif regularization_function == RegularizationFunction.L1:
            self.regularization_function = self.l1_regularization_func
        elif regularization_function == RegularizationFunction.HOYER_SQUARE_AND_L1_AND:
            self.regularization_function = self.hoyer_square_and_l1_regularization_func
        else:
            raise Exception("Unknown regularization function.")

        self.optimizer = None

        self.batch_size = dimension
        self.data = SigmaPiModelDataSet(function, dimension)
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        self.num_batch = numpy.ceil(len(self.data)/self.batch_size).astype(int)

        self.number_of_epochs = 200 * self.two_to_power_dimension
        self.number_of_fine_tune_epochs = int(self.number_of_epochs/8)

        self.gradient_change_func = None

        self.regularization_strength = regularization_strength

        self.log_interval = 50

        self.prune_limit = 0.01

        self.debug = False

    def disable_regularization(self):
        self.set_regularization_strength(0)

    def set_regularization_strength(self, regularization_strength):
        self.regularization_strength = regularization_strength

    def new_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def set_regularization_func(self, function):
        self.regularization_function = function

    def set_gradient_change_func(self, function):
        self.gradient_change_func = function

    def exponential_error(self, output, target):
        error = torch.exp((-output * target).sum())
        return error.clone()

    def relu_error(self, output, target):
        error = torch.relu(torch.tensor(1).to(self.device) - (output * target).sum())
        return error.clone()

    def hoyer_square_regularization_func(self):
        reg = 0.0
        for param in self.model.parameters():
            if param.requires_grad and torch.sum(torch.abs(param)) > 0:
                reg += (torch.sum(torch.abs(param)) ** 2) / torch.sum(param ** 2)

        return reg

    def l1_regularization_func(self):
        reg = 0.0
        for param in self.model.parameters():
            if param.requires_grad and torch.sum(torch.abs(param)) > 0:
                reg += torch.sum(torch.abs(param))

        return reg

    def hoyer_square_and_l1_regularization_func(self):
        return self.hoyer_square_regularization_func() + self.l1_regularization_func()

    def zero_out_gradients(self):
        for name, p in self.model.named_parameters():
            if 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            grad_tensor = p.grad.cpu().numpy()
            grad_tensor = numpy.where(tensor == 0, 0, grad_tensor)
            p.grad = torch.from_numpy(grad_tensor).to(self.device)

    def train(self, number_of_epochs):
        self.new_optimizer()
        self.model.train()
        for epoch in range(number_of_epochs):
            loss_value = 0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.reshape([target.size()[0], 1])

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_function(output, target)
                total_loss = loss
                reg = 0.0

                if self.regularization_strength != 0 and self.regularization_function is not None:
                    reg = self.regularization_strength * self.regularization_function()
                    total_loss += reg

                total_loss.backward()

                if self.gradient_change_func is not None:
                    self.gradient_change_func()

                self.optimizer.step()

                loss_value += loss.item()

            if epoch % self.log_interval == 0 and self.debug:
                percentage = 100 * epoch / number_of_epochs
                print(f'Train Epoch: {epoch} [({percentage:3.0f}%)] 'f'Loss: {loss_value/self.num_batch:.3f}  '
                      f'Reg: {reg:.3f}')

    def prune(self):
        for name, p in self.model.named_parameters():
            if 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            new_mask = numpy.where(abs(tensor) < self.prune_limit, 0, tensor)
            p.data = torch.from_numpy(new_mask).to(self.device)

    def test(self):
        self.model.eval()
        num_target = 0
        num_correct = 0
        with torch.no_grad():
            for data, target in self.data_loader:
                data = data.to(self.device)

                output = self.model(data)
                output_size = output.size()[0]
                output = output.reshape(output_size).tolist()
                target = target.tolist()

                for counter in range(output_size):
                    if numpy.sign(output[counter]) == target[counter]:
                        num_correct += 1
                    num_target += 1

        return num_correct, num_target

    def operation(self, debug=False):
        self.debug = debug

        self.set_gradient_change_func(None)
        self.train(self.number_of_epochs)

        self.prune()

        self.disable_regularization()
        self.set_gradient_change_func(self.zero_out_gradients)
        self.train(self.number_of_fine_tune_epochs)

        correct, target = self.test()
        return correct == target

    def zeroed_weights(self):
        for name, p in self.model.named_parameters():
            if 'mask' in name:
                continue
            return numpy.where(p.data.cpu().numpy() == 0)[1]

    def parameters(self):
        parameters = {"prune_limit": str(self.prune_limit), "simple_model": str(self.simple_model),
                      "decay": str(self.regularization_strength), "number_of_epochs": str(self.number_of_epochs),
                      "loss_function": str(self.loss_function_enum), "regularization_func":
                          str(self.regularization_function_enum)}

        return parameters

