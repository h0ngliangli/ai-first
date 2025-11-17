import numpy as np
import math
import matplotlib.pyplot as plt
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

a, b, c, d = np.random.randn(4)

learning_rate = 1e-6
for t in range(3000):
    y_pred = a + b * x + c * x**2 + d * x**3
    # Sum of Squared Errors (SSE) 平方误差和.
    # 和方差(Variance)不同，方差是均方误差(MSE)的无偏估计
    loss = np.square(y_pred - y).sum() # Sum of squared errors (SSE) 平方误差和
    # if t % 100 == 99:   
    #     print(t, loss, a, b, c, d)
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')


# plotting y = sin(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='sin(x)', color='blue', alpha=0.5)
plt.plot(x, y_pred, label='fit', color='red', alpha=0.7)
plt.legend()
plt.grid(True)
plt.show()
