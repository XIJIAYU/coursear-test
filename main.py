from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy
# 加载sklearn自带的mnist数据
digits = datasets.load_digits()
# 数据集包含1797个手写体数字的图片，图片大小为8*8
# 数字大小0～10，也就是说有这是个10分类问题
images = digits.images
targets = digits.target
print(("dataset shape is: "), images.shape)
# 将数据分为训练数据和测试数据（20%）
X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=0)
num_training = 1137
num_validation = 300
num_test = y_test.shape[0]
# 将训练集再分为训练集和验证集
mask = list(range(num_training, num_training + num_validation))
X_val = X_train[mask]
y_val = y_train[mask]
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]
mask = list(range(num_test))

print("the number of train: ", num_training)
print("the number of test: ", num_test)
print("the number of validation: ", num_validation)
# 将每个数字8*8的像素矩阵转化为64*1的向量
X_train = X_train.reshape(num_training, -1)
X_val = X_val.reshape(num_validation, -1)
X_test = X_test.reshape(num_test, -1)
print("training data shape: ", X_train.shape)
print("validation data shape: ", X_val.shape)
print("test data shape: ", X_test.shape)
# 定义神经网络的参数
# 定义超参
input_size = 64
hidden_size = 30
num_classes = 10
# 为了之后使用的方便，我将参数初始化，计算loss，训练，预测的过程都定义在一个名为network的类中
import numpy as np



class network(object):
    # 初始化参数,将W,b保存在名为params的字典中
    # W随机初始化，b初始化为零
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 定义损失函数，里面包含了前向传播的实现过程
    def loss(self, X, y=None, reg=0.0):
        # 先讲各个参数都提取出来
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        N, D = X.shape
        # 前向传播
        # hidden的实现
        hidden = np.dot(X, W1) + b1
        # relu:max(0, x)
        hidden = np.maximum(0, hidden)
        # 算输出y
        y2 = np.dot(hidden, W2) + b2
        # if y == None:
        #  return y2
        # loss 计算
        loss = None
        loss = -y2[range(N), y].sum() + np.log(np.exp(y2).sum(axis=1)).sum()
        loss = loss / N + 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        # 反向传播
        # 首先定义一个grads的字典，存放各个可训练参数的梯度
        grads = {}
        # 按照计算图，先计算dscore
        # 先对y2取对数
        exp = np.exp(y2)
        # 求每行个元素的和，之后用每行各个元素除上该行的和
        dscore = exp / exp.sum(axis=1, keepdims=True)
        # 对label（即y）对应的元素减1
        dscore[range(N), y] -= 1
        # 别忘了还要除输入样本的个数
        dscore = dscore / N
        grads['b2'] = np.sum(dscore, axis=0)
        grads['W2'] = np.dot(hidden.T, dscore) + reg * W2
        # dhidden
        dhidden = np.dot(dscore, W2.T)
        # 因为加了relu激活函数，随意要讲XW1 + b1 <0对应的dihidden元素归0
        dhidden[(np.dot(X, W1) + b1) < 0] = 0
        grads['b1'] = np.sum(dhidden, axis=0)
        grads['W1'] = np.dot(X.T, dhidden) + reg * W1
        return loss, grads

    # 训练神经网络，使用了随机梯度下降，和学习率衰减的技巧
    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        # 查看有有多少个训练样本，并检查按照设定的batch大小每个epoch需要迭代多少次
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # 使用随机梯度下降优化可训练参数
        # 把训练过程中得到的loss和准确率信息存起来方便查看并解决问题
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        # 迭代numz_iters次，每次只随机选择一个batch来训练样本
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]
            # 用当前的batch训练数据来得到loss 和grad
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            # 记录这次迭代的损失大小
            loss_history.append(loss)
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            # 如果你选择了可视化训练过程，那么会显示每次迭代产生的loss
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # 每个epoch结束，衰减一下学习率
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):

        y_pred = None
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
        scores = np.dot(hidden_layer, W2) + b2
        y_pred = np.argmax(scores, axis=1)
        return y_pred


net = network(input_size, hidden_size, num_classes)
stats = net.train(X_train, y_train, X_val, y_val,
                  num_iters=5000, batch_size=200,
                  learning_rate=0.01, learning_rate_decay=0.95,
                  reg=0.25, verbose=True)
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

test_acc = (net.predict(X_test) == y_test).mean()
print('test accuracy: ', test_acc)
