import  numpy
import matplotlib.pyplot as plt
import eg_neuralNetwork


from scipy.special import expit

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = eg_neuralNetwork.neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

with open("mnist_train - 副本.csv") as data_file:
    training_data_list = data_file.readlines()

# for n in range(8):
#     all_values = data_list[n].split(',')
#     image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
#     plt.imshow(image_array, cmap='Greys', interpolation='None')
#     plt.show()

epochs = 5

for e in range(epochs):
    for record in training_data_list:
        #以‘，’分开
        all_values = record.split(',')
        #将0-255的数缩放到  0.01-1.00
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01     #asfarray将字符串转为实数并创建数组
        #创建长度为  输出节点数量的一个数组
        targets = numpy.zeros(output_nodes) + 0.01
        #将标签对应的数改为0.99
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

with open("mnist_test_10.csv") as data_file:
    test_data_list = data_file.readlines()

scorecard = []


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

scorecard_array = numpy.asarray(scorecard)
print("数据正确率 = ", scorecard_array.sum() / scorecard_array.size)

