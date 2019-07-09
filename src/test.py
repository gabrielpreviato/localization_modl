import numpy as np
from matplotlib import pyplot as plt


x = np.random.normal(50, 50, 800)
y = np.random.normal(300, 50, 800)

x1 = np.random.normal(50, 150, 200)
y1 = np.random.normal(300, 150, 200)

x_s = np.append(x, x1)
y_s = np.append(y, y1)

plt.scatter(x_s, y_s)

x_p = np.mean(x_s)
y_p = np.mean(y_s)

plt.scatter(x_p, y_p)

plt.ylim(0, 600)
plt.xlim(0, 900)
plt.show()
