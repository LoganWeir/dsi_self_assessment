import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.formula.api as smf
import numpy as np

x = [0,1,2,3,5]
y = [1, 2.1, 3.2, 4, 6.1]

df = pd.DataFrame(list(zip(x, y)), columns=["x", "y"])

fig, ax = plt.subplots()

ax.scatter(df.x, df.y)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("X/Y Relationship")

# plt.show()

linear_model = smf.ols(formula='y ~ x', data=df)
relationship_model = linear_model.fit()

print relationship_model.params

x = np.linspace(0, 5, num=5)

ax.plot(x, relationship_model.params[0] + relationship_model.params[1] * x,
       linewidth=2, c="black")

plt.show()

print (relationship_model.params[0] + (relationship_model.params[1] * 4))

























# linear_model_summary(relationship_model)