import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class Classes:
    def __init__(self, num_class, num_stu, volume, num_stu_class_range, device):
        self.num_class = num_class
        self.volume = volume
        self.device = device
        self.num_selected = torch.zeros(num_class).to(device)
        self.set_num_selected(num_stu, num_stu_class_range[0], num_stu_class_range[1])
    
    def set_num_selected(self, num_stu, lowest_num, highest_num):
        # 生成每个课程的选课人数
        people = np.arange(lowest_num, highest_num+1)
        p = (people/num_stu)**(-0.2) * (1-people/num_stu)**(-0.2)
        p = p / np.sum(p)
        self.num_selected = torch.tensor(
            np.random.choice(
                people,
                self.num_class,
                replace=True,
                p=p
            )
        ).to(self.device)

class BaseNet(nn.Module):
    def __init__(self, device):
        super(BaseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        self.device = device
        self.to(device)
    
    def forward(self, x):
        x = self.net(x)
        return x

    def plot(self, highest_class_k=4):
        # 绘制k-豆函数图像
        x = torch.linspace(1, highest_class_k, 100).view(-1, 1).to(self.device)
        y = self(x)
        y[y<0] = 0
        fig, axs = plt.subplots()
        axs.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy())
        axs.set_xlabel("class_k")
        axs.set_ylabel("weight")
        axs.set_title("BaseNet")

class Students:
    def __init__(self, total_beans, num_stu, num_class, device):
        self.total_beans = total_beans
        self.num_stu = num_stu
        self.num_class = num_class
        self.base_net = BaseNet(device)
        self.given_beans = torch.zeros(num_stu, num_class).to(device)
        self.class_k = torch.zeros(num_class).to(device)
        self.device = device
    
    def select(self, classes: Classes):
        # 选择课程
        select_p = classes.num_selected / self.num_stu
        self.class_selected = torch.rand(self.num_stu, classes.num_class).to(self.device) < select_p

    def pricing(self, classes: Classes):
        # 计算每个学生的豆子数
        self.class_k = torch.tensor(classes.num_selected / classes.volume, dtype=torch.float32).to(self.device)  # (num_class,)
        # print("class_k: ", self.class_k)

        # 只对k>1的课程投豆
        not_hot_class = self.class_k <= 1
        hot_class_selected = self.class_selected.clone()  # (num_stu, num_class)
        hot_class_selected[:, not_hot_class] = False
        # print("hot_class_selected0: ", hot_class_selected[0, :])

        weights = hot_class_selected * self.base_net(self.class_k.reshape(-1, 1)).reshape(1, -1)  # (num_stu, num_class)
        weights[weights < 0] = 0
        weights = weights / (torch.sum(weights, axis=1, keepdims=True)+1e-5)  # (num_stu, num_class)
        self.given_beans = weights * self.total_beans  # (num_stu, num_class)

    def clean(self):
        # 清空变量
        self.class_selected = torch.zeros((self.num_stu, self.num_class)).to(self.device)
        self.given_beans = torch.zeros(self.num_stu, self.num_class).to(self.device)
        self.class_k = torch.zeros(self.num_class).to(self.device)
        self.beans_lower_bound = torch.zeros(self.num_class).to(self.device)

    def get_beans_lower_bound(self):
        # 计算每个课程的最低豆子数
        self.beans_lower_bound = torch.zeros(self.num_class).to(self.device)  # (num_class,)
        for i in range(self.num_class):
            if self.class_k[i] > 1:
                given_beans = self.given_beans[:, i][self.class_selected[:, i]]
                self.beans_lower_bound[i] = torch.quantile(
                    given_beans,
                    1-1/self.class_k[i]
                ).to(self.device)
        self.beans_lower_bound = torch.ceil(self.beans_lower_bound)
        return self.beans_lower_bound
    
    def plot_class_beans_dist(self):
        # 绘制三种课程的投豆频方图
        hot_class_k = torch.quantile(
            self.class_k,
            torch.tensor([0.5, 0.75, 1]).to(self.device),
            interpolation='nearest'
        ).to(self.device)
        fig, axs = plt.subplots(3)
        fig.set_size_inches(12, 10)
        fig.subplots_adjust(hspace=0.5)
        for i in range(3):
            hot_class_id = torch.argwhere(self.class_k == hot_class_k[i])[0].item()
            given_beans = self.given_beans[:, hot_class_id][self.class_selected[:, hot_class_id]].cpu().detach().numpy()
            axs[i].hist(given_beans, bins=100, range=(0, 100))
            axs[i].set_title(f"class_k: {hot_class_k[i]:.2f}, lower_bound: {self.beans_lower_bound[hot_class_id]}")
            axs[i].set_xlabel("beans")
            axs[i].set_ylabel("num_stu")
        fig.suptitle("Class beans distribution")