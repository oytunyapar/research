import torch
import numpy
from SignRepresentationNN.models import *
from SignRepresentationNN.data import *


class PruneSigmaPiModel:
    def __init__(self, function, dimension, simple_model=False):
        self.dimension = dimension
        self.two_to_power_dimension = 2 ** dimension
        self.total_number_of_functions = 2 ** self.two_to_power_dimension

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        if simple_model:
            self.model = SigmaPiSimpleModel(dimension).to(self.device)
            self.loss_function = self.exponential_error
        else:
            self.model = SigmaPiModel(dimension).to(self.device)
            self.loss_function = torch.nn.functional.mse_loss

        self.optimizer = None

        self.batch_size = dimension
        self.data = SigmaPiModelDataSet(function, dimension)
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        self.num_batch = numpy.ceil(len(self.data)/self.batch_size).astype(int)

        self.number_of_epochs = 400 * self.two_to_power_dimension
        self.number_of_fine_tune_epochs = int(self.number_of_epochs/4)

        self.regularization_func = self.hoyer_regularization_func
        self.gradient_change_func = None

        self.log_interval = 10

        self.prune_limit = 0.01

    def new_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def set_regularization_func(self, function):
        self.regularization_func = function

    def set_gradient_change_func(self, function):
        self.gradient_change_func = function

    def exponential_error(self, output, target):
        error = torch.exp((-output * target).sum())
        return error.clone()

    def hoyer_regularization_func(self):
        reg = 0.0
        for param in self.model.parameters():
            if param.requires_grad and torch.sum(torch.abs(param)) > 0:
                reg += (torch.sum(torch.abs(param)) ** 2) / torch.sum(param ** 2)

        return reg

    def zero_out_gradients(self):
        for name, p in self.model.named_parameters():
            if 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            grad_tensor = p.grad.cpu().numpy()
            grad_tensor = numpy.where(tensor == 0, 0, grad_tensor)
            p.grad = torch.from_numpy(grad_tensor).to(self.device)

    def train(self, number_of_epochs, decay=0.05):
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

                if decay != 0:
                    reg = decay * self.regularization_func()
                    total_loss += reg

                total_loss.backward()

                if self.gradient_change_func is not None:
                    self.gradient_change_func()

                self.optimizer.step()

                loss_value += loss.item()

            if epoch % self.log_interval == 0:
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

    def operation(self, decay=0.05):
        self.set_gradient_change_func(None)
        self.train(self.number_of_epochs, decay)
        correct, target = self.test()
        print(correct, "/", target)

        self.prune()

        self.set_gradient_change_func(self.zero_out_gradients)
        self.train(self.number_of_fine_tune_epochs, 0)
        correct, target = self.test()
        print(correct, "/", target)

    def zeroed_weights(self):
        for name, p in self.model.named_parameters():
            if 'mask' in name:
                continue
            return numpy.where(p.data.cpu().numpy() == 0)[1]
