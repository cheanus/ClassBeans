import torch
from school import *
import torch.optim as optim

def init_basenet(base_net: BaseNet, num_train, device):
    # 初始化base_net为直线函数
    x = torch.linspace(1, 4, 100).view(-1, 1).to(device)
    y = 2*(x-1)
    optimizer = optim.Adam(base_net.parameters(), lr=1e-2)
    for i in range(num_train):
        optimizer.zero_grad()
        output = base_net(x)
        loss = torch.mean((output - y) ** 2)
        loss.backward()
        optimizer.step()

def train():
    # 初始化参数
    num_class = 100
    volume = 50
    num_stu = 1000
    total_beans = 100
    num_train = 100
    lr = 1e-3
    lowest_num_stu_class = 10
    highest_num_stu_class = num_stu//10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = Classes(num_class, num_stu, volume, (lowest_num_stu_class, highest_num_stu_class), device)
    students = Students(total_beans, num_stu, num_class, device)
    init_basenet(students.base_net, 100, device)
    optimizer = optim.Adam(students.base_net.parameters(), lr=lr)

    for idx in range(num_train):
        # 清空数据
        students.clean()

        # 选择课程
        students.select(classes)
        # 计算每个学生的豆子数
        students.pricing(classes)

        # 计算每个课程的最低豆子数
        beans_lower_bound = torch.tensor(students.get_beans_lower_bound())
        # 计算loss
        loss1 = (students.given_beans - students.class_selected*beans_lower_bound.reshape(1, -1))
        loss1[loss1<=0] = -torch.log(-loss1[loss1<=0]+1)
        loss1 = torch.mean(loss1)
        x = torch.linspace(1, highest_num_stu_class/volume, 10).reshape(-1, 1).to(device)
        loss2 = (students.base_net(x).mean()-1)**2
        loss = loss1 + loss2

        # 更新参数
        optimizer.zero_grad()
        print(f"iter {idx}, loss1: {loss1}, loss2: {loss2}, loss: {loss}")
        loss.backward()
        optimizer.step()

    # 绘图
    students.plot_class_beans_dist()
    students.base_net.plot(highest_num_stu_class/volume)
    plt.show()

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 14})
    train()