import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.formula.api as smf
import numpy as np

x_y = [(0, 42.0), (1, -101.0), (2, 21.0), (3, -38.0), (4, 5.0), (7, 20.0), 
(9, 293.0), (11, 266.0), (15, 625.0), (20, 1266.0), (25, 1757.0), 
(30, 2844.0)]

df = pd.DataFrame(x_y, columns=["x", "y"])

fig, ax = plt.subplots()

ax.scatter(df.x, df.y)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("X/Y Relationship")

plt.show()

# linear_model = smf.ols(formula='y ~ x', data=df)
# relationship_model = linear_model.fit()

# # print relationship_model.params

# x = np.linspace(0, 30, num=30)

# ax.plot(x, relationship_model.params[0] + relationship_model.params[1] * x,
#        linewidth=2, c="black")

# plt.show()

# print (relationship_model.params[0] + (relationship_model.params[1] * 4))

























# linear_model_summary(relationship_model)