import numpy as np
from sklearn.ensemble import RandomForestRegressor
import GPU_forest as co2f
import matplotlib.pyplot as plt

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

rf = RandomForestRegressor(max_depth=4)

krf = co2f.GPUForestRegressor(C=3000, dual=False,tol = 0.001,max_iter=100000,kernel='linear',\
                                   max_depth=3,n_jobs=10,feature_ratio = 1.0,\
                                   n_estimators=100)

lw = 2

kernel_label = ["RF", "KRF"]
model_color = ["m", "c", "g"]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)

axes[0].plot(
    X,
    rf.fit(X, y).predict(X),
    color=model_color[0],
    lw=lw,
    label="{} model".format(kernel_label[0]),
)

axes[0].scatter(
    X,
    y,
    facecolor="none",
    edgecolor="k",
    s=50,
    label="other training data",
)


krf.fit(X, y)
axes[1].plot(
    X,
    krf.predict(X),
    color=model_color[0],
    lw=lw,
    label="{} model".format(kernel_label[0]),
)

axes[1].scatter(
    X,
    y,
    facecolor="none",
    edgecolor="k",
    s=50,
    label="other training data",
)


fig.text(0.5, 0.04, "data", ha="center", va="center")
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()