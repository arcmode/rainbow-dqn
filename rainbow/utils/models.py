import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialModelAggregator(nn.Module):
    def forward(self, outputs):
        # Distributional :
        # Value stream (batch, atoms)
        # Action stream (batch, actions, atoms)

        # Not Distributional (Classic)
        # Value stream (batch, )
        # Action stream (batch, actions)
        outputs = outputs["value"].unsqueeze(1) + outputs["actions"] - outputs["actions"].mean(dim=1, keepdim=True)
        return outputs


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma / self.in_features ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma / self.out_features ** 0.5)

    def forward(self, input):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(input, weight, bias)


class ModelBuilder():
    def __init__(self, units, dropout, nb_states, nb_actions, l2_reg, window, distributional, nb_atoms, adversarial, noisy):
        self.units = units
        self.dropout = dropout
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.l2_reg = l2_reg
        self.recurrent = (window > 1)
        self.window = window
        self.distributional = distributional
        self.nb_atoms = nb_atoms
        self.adversarial = adversarial
        self.noisy = noisy

    def dense(self, in_features, out_features):
        if self.noisy:
            return NoisyLinear(in_features, out_features, sigma=0.5)
        return nn.Linear(in_features, out_features)

    def build_model(self, trainable=True):
        class Net(nn.Module):
            def __init__(self, builder):
                super(Net, self).__init__()
                self.builder = builder
                self.recurrent = builder.recurrent
                self.distributional = builder.distributional
                self.adversarial = builder.adversarial
                self.nb_actions = builder.nb_actions
                self.nb_atoms = builder.nb_atoms
                self.dropout = builder.dropout
                self.l2_reg = builder.l2_reg
                self.noisy = builder.noisy

                if self.recurrent:
                    self.lstm_layers = nn.ModuleList()
                    input_dim = builder.nb_states
                    for i, units in enumerate(builder.units):
                        lstm = nn.LSTM(input_size=input_dim, hidden_size=units, batch_first=True)
                        self.lstm_layers.append(lstm)
                        input_dim = units
                else:
                    layers = []
                    input_dim = builder.nb_states
                    for units in builder.units:
                        layers.append(self.builder.dense(input_dim, units))
                        layers.append(nn.ReLU())
                        if self.dropout > 0:
                            layers.append(nn.Dropout(self.dropout))
                        input_dim = units
                    self.main_stream = nn.Sequential(*layers)

                if self.distributional and self.adversarial:
                    self.action_stream_fc1 = nn.Linear(input_dim, 512)
                    self.action_stream_fc2 = self.builder.dense(512, self.builder.nb_atoms * self.nb_actions)
                    self.value_stream_fc1 = nn.Linear(input_dim, 512)
                    self.value_stream_fc2 = self.builder.dense(512, self.builder.nb_atoms)
                    self.output_activation = nn.Softmax(dim=-1)

                elif self.distributional and not self.adversarial:
                    self.output_layer = self.builder.dense(input_dim, self.builder.nb_atoms * self.nb_actions)
                    self.output_activation = nn.Softmax(dim=-1)

                elif not self.distributional and self.adversarial:
                    self.action_stream_fc1 = nn.Linear(input_dim, 256)
                    self.action_stream_fc2 = self.builder.dense(256, self.nb_actions)
                    self.value_stream_fc1 = nn.Linear(input_dim, 256)
                    self.value_stream_fc2 = self.builder.dense(256, 1)

                else:
                    self.output_layer = nn.Linear(input_dim, self.nb_actions)

            def forward(self, x):
                if self.recurrent:
                    h = x
                    for i, lstm in enumerate(self.lstm_layers):
                        h, _ = lstm(h)
                        if i + 1 != len(self.lstm_layers):
                            pass
                        else:
                            h = h[:, -1, :]
                else:
                    h = self.main_stream(x)

                if self.distributional and self.adversarial:
                    action_stream = F.relu(self.action_stream_fc1(h))
                    if self.dropout > 0:
                        action_stream = F.dropout(action_stream, p=self.dropout, training=self.training)
                    action_stream = self.action_stream_fc2(action_stream)
                    action_stream = action_stream.view(-1, self.nb_actions, self.nb_atoms)

                    value_stream = F.relu(self.value_stream_fc1(h))
                    if self.dropout > 0:
                        value_stream = F.dropout(value_stream, p=self.dropout, training=self.training)
                    value_stream = self.value_stream_fc2(value_stream)

                    outputs = {"value": value_stream, "actions": action_stream}
                    output = AdversarialModelAggregator()(outputs)
                    output = self.output_activation(output)

                elif self.distributional and not self.adversarial:
                    output = self.output_layer(h)
                    output = output.view(-1, self.nb_actions, self.nb_atoms)
                    output = self.output_activation(output)

                elif not self.distributional and self.adversarial:
                    action_stream = F.relu(self.action_stream_fc1(h))
                    if self.dropout > 0:
                        action_stream = F.dropout(action_stream, p=self.dropout, training=self.training)
                    action_stream = self.action_stream_fc2(action_stream)

                    value_stream = F.relu(self.value_stream_fc1(h))
                    if self.dropout > 0:
                        value_stream = F.dropout(value_stream, p=self.dropout, training=self.training)
                    value_stream = self.value_stream_fc2(value_stream)

                    outputs = {"value": value_stream.squeeze(-1), "actions": action_stream}
                    output = AdversarialModelAggregator()(outputs)[:, 0, :]

                else:
                    output = self.output_layer(h)

                return output

        model = Net(self)
        return model
