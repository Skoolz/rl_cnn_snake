import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class DQN(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)

        self.fc_adv1 = nn.Linear(576, 512)
        self.fc_adv2 = nn.Linear(512, outputs)

        self.fc_v = nn.Linear(576,1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.fc_adv1(x))
        adv = self.fc_adv2(adv)

        v = F.relu(self.fc_v(x))
        

        return v-adv-torch.mean(adv)


class QTrainer:
    def __init__(self, model:nn.Module,target, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target = target
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_state))
        )
        final_mask = ~non_final_mask

        non_final_next_states = torch.cat(
            [s for s in next_state if s is not None]
        )
        # 1: predicted Q values with current state
        state_action_values = self.model(state).gather(1, action.unsqueeze(1))
        
        expected_state_action_values = torch.zeros(64)
        expected_state_action_values[non_final_mask] = (
                self.target(non_final_next_states).max(1)[0].detach() * self.gamma
                + reward[non_final_mask].detach()
        )
        expected_state_action_values[final_mask] = reward[final_mask].detach()
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(expected_state_action_values.unsqueeze(1), state_action_values)
        #for param in self.model.parameters():
        #        param.grad.data.clamp_(-1, 1)
        loss.backward()

        self.optimizer.step()



