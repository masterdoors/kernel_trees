import numpy as np
from sklearn.ensemble import RandomForestRegressor
from kernel_trees import CO2ForestRegressor
from kernel_trees import RefinedForestRegressor
from deepforest import CascadeForestRegressor
import matplotlib.pyplot as plt

X = np.sort(5 * np.random.rand(400, 1), axis=0)
y = np.sin(X).ravel()

# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(80))

est = [RandomForestRegressor(max_depth=4) for _ in range(2)]

model = CascadeForestRegressor(max_layers=3)
model.set_estimator(est)  


lw = 2

kernel_label = ["RF", "Cascade KRF","Reg Cascade KRF"]
model_color = ["m", "c", "g"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)

model.fit(X, y)
axes[0].plot(
    X,
    model.predict(X),
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


est = [CO2ForestRegressor(C=3000, dual=False,tol = 0.001,max_iter=1000000,kernel='linear',\
                                   max_depth=3,n_jobs=10,feature_ratio = 0.5,\
                                   n_estimators=100) for i in range(int(2))]
                                
model = CascadeForestRegressor(max_layers=3)
model.set_estimator(est)  


model.fit(X, y)
axes[1].plot(
    X,
    model.predict(X),
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

est = [RefinedForestRegressor(C=3000, dual=False,tol = 0.001,max_iter=1000000,kernel='linear',\
                                   max_depth=3,n_jobs=10,feature_ratio = 0.5,\
                                   n_estimators=100, prune_threshold=0.1, pruneC=10000.0) for i in range(int(2))]
                                
model = CascadeForestRegressor(max_layers=3)
model.set_estimator(est)  


model.fit(X, y)
axes[2].plot(
    X,
    model.predict(X),
    color=model_color[0],
    lw=lw,
    label="{} model".format(kernel_label[0]),
)

axes[2].scatter(
    X,
    y,
    facecolor="none",
    edgecolor="k",
    s=50,
    label="other training data",
)


fig.text(0.5, 0.04, "data", ha="center", va="center")
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
fig.suptitle("Cascade (deep) Forest: RFR vs KFR vs refined KFR", fontsize=14)
plt.show()