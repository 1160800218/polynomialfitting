#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import math

# 梯度下降法
def gradientDescent(A, x, b, alpha):
    epsilon = 1e-3
    grad = np.dot(A, x) - b                             # 梯度grad(k)
    k = 1
    while not np.linalg.norm(grad, ord=2) < epsilon:
        x = x - alpha * grad                            # 沿负梯度方向搜索x(k+1)=x-alpha*grad(k)
        grad = np.dot(A, x) - b                         # grad(k+1)=A*x(k+1)-b
        k += 1                                          # 迭代次数
    print '梯度下降法的迭代次数为：', k, '次'
    return x

# 共轭梯度法
def conjugateGradient(A, x, b):
    epsilon = 1e-4
    r = b - np.dot(A, x)                                # 残差r(0)=b=A*x(0)
    p = r                                               # p(0)=r(0)
    for k in range(len(b)):
        alpha = np.dot(r.T, r) / np.dot(p.T, A).dot(p)  # 学习率alpha(k),标量
        x = x + alpha * p                               # x(k+1)=x(k)+alpha(k)*p(k)
        r_ = r - alpha * np.dot(A, p)                    # r(k+1)=r(k)-alpha(k)*A*p(k)
        if np.linalg.norm(p, ord=2) < epsilon:          # ||p(k)||是否足够小
            print '共轭梯度法的迭代次数为：', k, '次'
            return x
        beta = np.dot(r_.T, r_) / np.dot(r.T, r)           # beta(k)
        p = r_ + beta * p                                  # p(k+1)=r(k+1)+beta(k)*p(k）
        r = r_
    print '共轭梯度法的迭代次数为：', len(b), '次'
    return x

# 多项式拟合
class PolynomialFitting(object):
    def __init__(self, degree, scale):
        self.degree = degree
        self.scale = scale
        self.train_scale = int(scale / 2)
        self.valid_scale = int((scale - self.train_scale) / 2)
        self.test_scale = int(scale - self.train_scale - self.valid_scale)
    # 生成样本集合
    def generatingX(self):
        self.x = np.arange(-1.0, 1.0, 2.0/self.scale)
        self.train_x = np.ones(self.train_scale)      # 训练集合
        self.valid_x = np.ones(self.valid_scale)      # 验证集合
        self.test_x = np.ones(self.test_scale)        # 测试集合
        k = 0
        while k < self.valid_scale:
            self.train_x[k] = self.x[2*k]
            self.valid_x[k] = self.x[2*k+1]
            k += 1
        while k < self.train_scale:
            self.train_x[k] = self.x[2*k]
            self.test_x[k - self.valid_scale] = self.x[2*k+1]
            k += 1
        if 2 * k != self.scale:
            self.test_x[self.test_scale-1] = self.x[2*k-1]

        self.train_X = np.ones((self.train_scale, self.degree+1))
        self.valid_X = np.ones((self.valid_scale, self.degree + 1))
        self.test_X = np.ones((self.test_scale, self.degree + 1))
        for i in range(self.train_scale):
            for j in range(self.degree + 1):
                self.train_X[i, j] = math.pow(self.train_x[i], j)
                if i < self.valid_scale:
                    self.valid_X[i, j] = math.pow(self.valid_x[i], j)
                if i < self.test_scale:
                    self.test_X[i, j] = math.pow(self.test_x[i], j)

    # 生成含0均值的高斯噪声的目标值t集合
    def generatingT(self):
        noise = np.random.normal(loc=0.0, scale=0.01, size=len(self.x))                 # 0均值，0.01标准差的噪声
        self.T = (np.array(np.sin(self.x * np.pi * 2)) + noise).reshape(self.scale, 1)
        self.train_T = np.ones(self.train_scale).reshape(self.train_scale, 1)           # 训练目标值
        self.valid_T = np.ones(self.valid_scale).reshape(self.valid_scale, 1)           # 验证目标值
        self.test_T = np.ones(self.test_scale).reshape(self.test_scale, 1)              # 测试目标值
        k = 0
        while k < self.valid_scale:
            self.train_T[k] = self.T[2 * k]
            self.valid_T[k] = self.T[2 * k + 1]
            k += 1
        while k < self.train_scale:
            self.train_T[k] = self.T[2 * k]
            self.test_T[k - self.valid_scale] = self.T[2 * k + 1]
            k += 1
        if 2 * k != self.scale:
            self.test_T[self.test_scale - 1] = self.T[2 * k - 1]

    # 解析解求系数
    def calculateW(self, regular):
        self.W = np.linalg.inv(np.dot(self.train_X.T, self.train_X)).dot(self.train_X.T).dot(self.train_T)
        self.W_regular = 2 * np.linalg.inv(2 * np.dot(self.train_X.T, self.train_X) + regular * np.eye(self.train_X.shape[1])).dot(self.train_X.T).dot(self.train_T)
        # print self.W
        # print self.W_regular

    # 梯度下降、共轭梯度法求系数
    def calculateWWithGrad(self, regular, method):
        w = np.ones(self.degree + 1)
        w = w.reshape(self.degree + 1, 1)
        A = (2 * np.dot(self.train_X.T, self.train_X) + regular * np.eye(self.train_X.shape[1])) / self.train_scale         # 正定矩阵A
        b = 2 * np.dot(self.train_X.T, self.train_T) / self.train_scale                                                     # 目标矩阵b
        if method == '梯度下降法':
            self.W_gd = gradientDescent(A, w, b, 0.1)
        if method == '共轭梯度法':
            self.W_cg = conjugateGradient(A, w, b)

    # 求损失函数
    def loss(self):  # 参数pf为PolynomialFitting类对象
        loss = (np.dot(self.valid_X, self.W) - self.valid_T).T.dot(np.dot(self.valid_X, self.W) - self.valid_T)[
                   0, 0] / self.valid_scale
        return loss
    def lossWithRegular(self, regular, W):
        loss = ((np.dot(self.valid_X, W) - self.valid_T).T.dot(
            np.dot(self.valid_X, W) - self.valid_T)
                + regular * np.linalg.norm(W, ord=2) ** 2 / 2)[0, 0] / self.valid_scale
        return loss

    # 获得最优超参数（惩罚系数）
    def getBestRegular(self, rs, method):
        loss = 65535
        minloss = 65535
        self.bestr = 0
        W = np.ones(self.degree + 1)
        self.bestW = np.ones(self.degree + 1)               # 惩罚项取最优时的系数向量
        for r in rs:
            if method == '解析法':
                PolynomialFitting.calculateW(self, r)
                loss = PolynomialFitting.lossWithRegular(self, r, self.W_regular)
                W = self.W_regular
            if method == '梯度下降法':
                PolynomialFitting.calculateWWithGrad(self, r, method)
                loss = PolynomialFitting.lossWithRegular(self, r, self.W_gd)
                W = self.W_gd
            if method == '共轭梯度法':
                PolynomialFitting.calculateWWithGrad(self, r, method)
                loss = PolynomialFitting.lossWithRegular(self, r, self.W_cg)
                W = self.W_cg
            if loss < minloss:
                minloss = loss
                self.bestr = r
                self.bestW = W
            print loss
        print '惩罚系数取', self.bestr, '时代价最低，最优代价为：', minloss

    def fitting(self, regulars, method):
        # x = np.arange(-1.0, 1.0, 0.001)
        # y = np.sin(2 * np.pi * x)
        # ax[0][0].plot(x, y)
        PolynomialFitting.getBestRegular(self, regulars, method)
        # if method == '解析法':
        #     ax[0][0].plot(self.train_x, self.train_X.dot(self.W), 'b.-')
        #     ax[0][1].plot(self.train_x, self.train_X.dot(self.bestW), 'r+-')
        # if method == '梯度下降法':
        #     ax[1][0].plot(self.train_x, self.train_X.dot(self.bestW), 'g*-')
        # if method == '共轭梯度法':
        #     ax[1][1].plot(self.train_x, self.train_X.dot(self.bestW), 'yo-')
        # plt.plot(self.train_x, self.train_T, 'g.')



def main():
    p = PolynomialFitting(degree=9, scale=30)
    p.generatingX()
    p.generatingT()
    regulars = [0.00001, 0.0001, 0.001]
    methods = ['解析法', '梯度下降法', '共轭梯度法']
    for method in methods:
        p.fitting(regulars, method)

    fig1 = plt.figure()
    fig1.suptitle("degree=" + str(9))
    ax = fig1.subplots(2, 2)
    ax[0][0].plot(p.train_x, p.train_X.dot(p.W), 'b.-')
    ax[0][1].plot(p.train_x, p.train_X.dot(p.bestW), 'r+-')
    ax[1][0].plot(p.train_x, p.train_X.dot(p.bestW), 'g*-')
    ax[1][1].plot(p.train_x, p.train_X.dot(p.bestW), 'yo-')

    plt.show()

if __name__ == "__main__":
    main()
