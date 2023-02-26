import torch
import numpy
from SignRepresentationNN.models import *
from SignRepresentationNN.data import *


class PruneSigmaPiModel:
    def __init__(self, function, dimension):
        self.dimension = dimension
        self.two_to_power_dimension = 2 ** dimension
        self.total_number_of_functions = 2 ** self.two_to_power_dimension

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = SigmaPiModel(dimension).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.batch_size = dimension
        self.data = SigmaPiModelDataSet(function, dimension)
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

        self.number_of_epochs = 200 * self.two_to_power_dimension
        self.number_of_fine_tune_epochs = int(self.number_of_epochs/4)

        self.log_interval = 50

        self.decay = 0.05
        self.prune_limit = 0.01

    def new_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self):
        self.model.train()
        for epoch in range(self.number_of_epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.reshape([target.size()[0], 1])

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = torch.nn.functional.mse_loss(output, target)

                reg = 0.0
                for param in self.model.parameters():
                    if param.requires_grad and torch.sum(torch.abs(param)) > 0:
                        reg += (torch.sum(torch.abs(param)) ** 2) / torch.sum(param ** 2)

                total_loss = loss + self.decay * reg
                total_loss.backward()

                self.optimizer.step()

                if batch_idx == 0 and epoch % self.log_interval == 0:
                    percentage = 100 * epoch / self.number_of_epochs
                    print(f'Train Epoch: {epoch} [({percentage:3.0f}%)] 'f'Loss: {loss.item():.3f}  Reg: {reg:.3f}')

    def fine_tune_train(self):
        self.new_optimizer()
        self.model.train()
        for epoch in range(self.number_of_fine_tune_epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.reshape([target.size()[0], 1])

                self.optimizer.zero_grad()

                output = self.model(data)
                loss = torch.nn.functional.mse_loss(output, target)
                loss.backward()

                for name, p in self.model.named_parameters():
                    if 'mask' in name:
                        continue
                    tensor = p.data.cpu().numpy()
                    grad_tensor = p.grad.cpu().numpy()
                    grad_tensor = numpy.where(tensor == 0, 0, grad_tensor)
                    p.grad = torch.from_numpy(grad_tensor).to(self.device)

                self.optimizer.step()

                if batch_idx == 0 and epoch % self.log_interval == 0:
                    percentage = 100 * epoch / self.number_of_fine_tune_epochs
                    print(f'Fine Tune Train Epoch: {epoch} [({percentage:3.0f}%)] 'f'Loss: {loss.item():.3f}')

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

    def operation(self):
        self.train()
        correct, target = self.test()
        print(correct, "/", target)

        self.prune()
        self.fine_tune_train()
        correct, target = self.test()
        print(correct, "/", target)

    def zeroed_weights(self):
        for name, p in self.model.named_parameters():
            if 'mask' in name:
                continue
            return numpy.where(p.data.cpu().numpy() == 0)[1]
