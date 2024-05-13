import numpy.random
import scipy.special


class neuralNetwork:

    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 权重隐藏层与输入层之间的权重   为self.hnodes  *  self.inodes大小的矩阵   W  *  I  = H   I为列向量
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))     #以0.0为中心正态分布随机值
        # 隐藏层与输出层之间的权重   为self.onodes  *  self.hnodes大小的矩阵   W  *  H  = O
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))     #边界为1/（传入链接数目）^(-0.5)
        #S激活函数
        self.activation_function = lambda x:scipy.special.expit(x)

    #训练函数
    def train(self,inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        #隐藏次输入    权重矩阵与输入矩阵点乘
        hidden_inputs = numpy.dot(self.wih, inputs)
        #隐藏层输出    激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        #输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #输出层输出
        final_outputs = self.activation_function(final_inputs)
        #计算输出误差
        output_errors = targets - final_outputs
        #隐藏层误差          WT  *  Oe
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #隐藏层输出层权重优化
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        #输入层隐藏层权重优化
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0- hidden_outputs)),
                                        numpy.transpose(inputs))


        pass


    #查询函数
    def query(self,inputs_list):

        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
