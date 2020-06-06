import numpy as np
import random
import matplotlib.pyplot as plt
import time


def load_data_set(filename):
    # 创建数据标签列表
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        for i in range(500):
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_x):
    # sigmoid 函数
    return 1.0 / (1 + np.exp(-in_x))


def grad_ascent(data_mat_in, class_labels):
    time0 = time.time()
    # 梯度上升优化算法
    # 将数据集和标签集转换为numpy 矩阵数据类型，且对标签矩阵进行转置，（1，m）->(m, 1)
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    # 得到数据集的大小，m行n列
    m, n = np.shape(data_matrix)
    # 设置移动步长，学习速率，控制更新的幅度
    alpha = 0.001
    # 最大迭代次数
    max_cycles = 500
    # 初始化w，n行1列的全1 矩阵数据集
    weights = np.ones((n, 1))
    # 迭代爬坡，利用数学公式计算最佳参数w， 矢量化公式：w := w + aX^T(y - g(Xw))
    for k in range(max_cycles):
        # 矩阵乘法（m, n）* (n, 1) = (m, 1),计算g(Xw)
        h = sigmoid(data_matrix * weights)
        error = label_mat - h
        # （n, m）* (m, 1) = (n, 1)得到n个参数，通过不断迭代优化参数
        weights = weights + alpha * data_matrix.transpose() * error
    print(f'梯度上升算法耗时： {time.time() - time0}')
    # 将weights转换为array
    return weights.getA()


def rand_grad_ascent(data_mat_in, class_labels, num_iter=20):
    # 随机梯度优化算法，降低算法复杂度，但稳定度较一般梯度算法有所下降\
    # 记录随机算法耗时
    time0 = time.time()
    data_matrix = np.array(data_mat_in)
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        # 内循环次数选取总数据的1/100
        inner_num = int(m / 100)
        # 每次只随机选取几组数据进行计算，比起整个矩阵运算，大大降低了时间复杂度
        for i in range(inner_num):
            # 控制学习速率，随着不断的爬坡，alpha的值适当减少以不断逼近极值
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            rand_choice = data_index[rand_index]
            # 公式计算步幅，更新参数
            error = class_labels[rand_choice] - sigmoid(sum(data_matrix[rand_choice] * weights))
            step = alpha * error * data_matrix[rand_choice]
            weights = weights + step
            # 删除以随机选取过的数据下标
            del(data_index[rand_index])
    print(f'随机梯度上升算法耗时： {time.time() - time0}')
    return weights


def grad(data_mat_in, class_labels):
    data_matrix = np.array(data_mat_in)
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmoid(np.sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights


def plot_best_fit(filename, weights):
    # 可视化数据集并利用最佳参数画出决策边界
    data_mat, label_mat = load_data_set(filename)
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    # 记录正样本
    x_cord1 = []
    y_cord1 = []
    # 记录负样本
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        # 1表示正样本， 0表示负样本， 第一列为x1， 第二列为x2
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    # 绘制图像
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # marker='s'， 区别于负样本，用正方形绘制
    # alpha=.5 表示透明度0.5
    ax.scatter(x_cord1, y_cord1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(x_cord2, y_cord2, s=20, c='green', alpha=.5)
    x = np.arange(-3.0, 3.0, 0.1)
    # 利用系数w得到分界线， w0 + w1x1 + w2x2 = 0 -> x2 = (-w0 -w1x1) / w2
    y = np.array((- weights[0] - weights[1] * x) / weights[2])
    ax.plot(x, y)
    plt.title('DataSet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


data, label = load_data_set('test_set')
w = grad_ascent(data, label)
plot_best_fit('test_set', w)
w1 = rand_grad_ascent(data, label, num_iter=50)
print(w, w1)
plot_best_fit('test_set', w1)
w2 = grad(data, label)
plot_best_fit('test_set', w2)
