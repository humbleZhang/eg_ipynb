import numpy
import matplotlib.pyplot as plt
import eg_neuralNetwork
from scipy.special import expit
from tqdm import tqdm

# 定义神经网络结构和学习率
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

# 创建神经网络实例
n = eg_neuralNetwork.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读取训练数据
with open("mnist_train - 副本.csv") as data_file:
    training_data_list = data_file.readlines()

# 定义训练的轮次
epochs = 5

# 训练神经网络
for e in range(epochs):
    print(f"Starting Epoch {e + 1}/{epochs}")

    # 使用 tqdm 创建进度条
    training_progress = tqdm(training_data_list, desc=f'Epoch {e + 1}', total=len(training_data_list))

    for record in training_progress:
        # 以‘，’分开
        all_values = record.strip().split(',')  # 确保去除可能的空白字符
        # 将标签转换为整数
        label = int(all_values[0])
        # 将剩余的字符串列表转换为浮点数数组，然后进行归一化
        inputs = (numpy.asarray([int(val) for val in all_values[1:]], dtype=float) / 255.0 * 0.99) + 0.01
        # 创建目标数组，并设置正确的标签值为0.99
        targets = numpy.zeros(output_nodes, dtype=float) + 0.01
        targets[label] = 0.99
        # 训练神经网络
        n.train(inputs, targets)

    training_progress.close()
    print(f"Epoch {e + 1}/{epochs} completed\n")

# 读取测试数据
with open("mnist_test_10.csv") as data_file:
    test_data_list = data_file.readlines()

# 初始化准确率记录列表
scorecard = []

# 使用神经网络进行测试数据的预测
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

# 计算准确率
scorecard_array = numpy.asarray(scorecard)
accuracy = scorecard_array.sum() / scorecard_array.size
print(f"Accuracy: {accuracy:.2%}")