import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
import numpy as np
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

# load data
samples_path = r"data/samples2.npy"
features_path = r"data/features2.npy"

# save parameters
model_path = r"data/models/mappingNN2"

X = np.load(features_path)

Y = np.load(samples_path) + 1
print(np.max(Y), np.min(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
#
# # KMeans
# n_clusters = 10
# kmeans = KMeans(n_clusters=n_clusters, verbose=1)
# kmeans.fit(X_test)
#
# pca = PCA(n_components=5)
# embedded = pca.fit_transform(X_test)
# plt.scatter(embedded[:, 0], embedded[:, 1], c=kmeans.labels_)
# plt.show()
# plt.scatter(embedded[:, 1], embedded[:, 2], c=kmeans.labels_)
# plt.show()
# plt.scatter(embedded[:, 3], embedded[:, 4], c=kmeans.labels_)
# plt.show()
#
# plt.scatter(Y_train[:, 0], Y_train[:, 1])
# plt.show()
# plt.scatter(Y_train[:, 2], Y_train[:, 3])
# plt.show()
# plt.scatter(Y_train[:, 4], Y_train[:, 5])
# plt.show()
#
# tsne = TSNE(n_components=2, perplexity=2, verbose=1)
# embedded = tsne.fit_transform(X_test)
# plt.scatter(embedded[:, 0], embedded[:, 1], c=kmeans.labels_)
# plt.show()

# # NEAREST NEIGHBOR
# Y_train = Y_train[:, 0]
# Y_test = Y_test[:, 0]
# print('Fitting NN')
# NN = KNeighborsRegressor(n_neighbors=10,
#                          weights="uniform", algorithm='auto', leaf_size=30, p=2, metric='minkowski'
#                          , metric_params=None, n_jobs=None)
#
# NN.fit(X_train, Y_train)
# print('NN Fitted')
#
# # best_model = clf.best_estimator_
# test_score = NN.score(X_test, Y_test)
# print(mean_squared_error(Y_test, NN.predict(X_test)))
# exit()
# NEURAL NETWORK
import keras
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(2048,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(256, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(inputs)
predictions = Dense(200)(x)

# This creates a iv3_model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])
# iv3_model.fit(X_train, Y_train, verbose=1, epochs=10)  # starts training

# launch TensorBoard (data won't show up until after the first epoch)
tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=False,
                                 write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None)

# fit the iv3_model with the TensorBoard callback
history = model.fit(
    X_train, Y_train,
    batch_size=64,
    epochs=20,
    verbose=1,
    callbacks=[tb],
    validation_split=0.2
)

model.save(model_path)

Y_pred = model.predict(X_test, batch_size=50)
for i in range(10):
    print(Y_pred[i])
    print(Y_pred[i] - Y_test[i])

# RANDOM FOREST
# Y_train = Y_train[:, 0]
# Y_test = Y_test[:, 0]
# print(X_train.shape)
# print(Y_train.shape)
# # xgb_model = MultiOutputRegressor(RandomForestRegressor(max_depth=10, n_estimators=50), n_jobs=2)
# xgb_model = RandomForestRegressor(max_depth=100, n_estimators=20)
# # print(xgb_model.get_params().keys())
# clf = GridSearchCV(xgb_model,
#                    {'max_depth': [10, 50, 100],
#                     'n_estimators': [10, 20]}, verbose=1, cv=3)
# print('Fitting')
# xgb_model.fit(X_train, Y_train)
# print('Fitted')
#
# # best_model = clf.best_estimator_
# test_score = xgb_model.score(X_test, Y_test)
# print(mean_squared_error(Y_test, xgb_model.predict(X_test)))
