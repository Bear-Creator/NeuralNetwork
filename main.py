import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Веса
        self.weights_input_hidden1 = np.random.randn(self.input_size, self.hidden_size1) * np.sqrt(2 / self.input_size)
        self.weights_hidden1_hidden2 = np.random.randn(self.hidden_size1, self.hidden_size2) * np.sqrt(2 / self.hidden_size1)
        self.weights_hidden2_output = np.random.randn(self.hidden_size2, self.output_size) * np.sqrt(2 / self.hidden_size2)
        
        # Смещения
        self.bias_hidden1 = np.zeros((1, self.hidden_size1))
        self.bias_hidden2 = np.zeros((1, self.hidden_size2))
        self.bias_output = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, x):
        self.hidden1 = self.sigmoid(np.dot(x, self.weights_input_hidden1) + self.bias_hidden1)
        self.hidden2 = self.sigmoid(np.dot(self.hidden1, self.weights_hidden1_hidden2) + self.bias_hidden2)
        self.output = self.sigmoid(np.dot(self.hidden2, self.weights_hidden2_output) + self.bias_output)
        return self.output
    
    def backward(self, x, y, output):
        error = y - output
        d_output = error * self.sigmoid_derivative(output)
        
        error_hidden2 = d_output.dot(self.weights_hidden2_output.T)
        d_hidden2 = error_hidden2 * self.sigmoid_derivative(self.hidden2)
        
        error_hidden1 = d_hidden2.dot(self.weights_hidden1_hidden2.T)
        d_hidden1 = error_hidden1 * self.sigmoid_derivative(self.hidden1)
        
        # Обновление весов и смещений
        self.weights_hidden2_output += self.hidden2.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_hidden1_hidden2 += self.hidden1.T.dot(d_hidden2) * self.learning_rate
        self.bias_hidden2 += np.sum(d_hidden2, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden1 += x.T.dot(d_hidden1) * self.learning_rate
        self.bias_hidden1 += np.sum(d_hidden1, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, x, y, epochs):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output)
            if epoch % 5000 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, x):
        return self.forward(x)

# Подготовка данных для нейронки
lower_to_upper = {chr(i): chr(i - 32) for i in range(97, 123)}
upper_to_lower = {chr(i): chr(i + 32) for i in range(65, 91)}

data = []
labels = []

for lower, upper in lower_to_upper.items():
    data.append(ord(lower))
    labels.append(ord(upper))

for upper, lower in upper_to_lower.items():
    data.append(ord(upper))
    labels.append(ord(lower))

data = np.array(data).reshape(-1, 1)
labels = np.array(labels).reshape(-1, 1)

data = data / 127
labels = labels / 127

# Параметры
input_size = 1
hidden_size1 = 4
hidden_size2 = 8
output_size = 18
learning_rate = 0.1
epochs = 200000

nn = SimpleNN(input_size, hidden_size1, hidden_size2, output_size, learning_rate)
nn.train(data, labels, epochs)

def convert_letter(letter, nn):
    letter_code = ord(letter) / 127.0
    predicted_code = nn.predict(np.array([[letter_code]]))[0][0] * 127
    return chr(int(round(predicted_code)))

# Тесты
def test_convert_letter(nn):
    test_cases = {
        'a': 'A',
        'b': 'B',
        'c': 'C',
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        'A': 'a',
        'B': 'b',
        'C': 'c',
        'X': 'x',
        'Y': 'y',
        'Z': 'z'
    }
    
    passed = 0
    for lower, upper in test_cases.items():
        result_upper = convert_letter(lower, nn)
        result_lower = convert_letter(upper, nn)
        
        if result_upper == upper and result_lower == lower:
            passed += 1
            print(result_upper, result_lower)
        else:
            print(f"Test failed for {lower} -> {result_upper} and {upper} -> {result_lower}")
    
    print(f"Passed {passed} out of {len(test_cases)} tests.")

# Тест
print(convert_letter('a', nn))
print(convert_letter('A', nn)) 

# БОЛЬШЕ ТЕСТОВ
test_convert_letter(nn)
