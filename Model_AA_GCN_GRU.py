import keras
import numpy as np
import spektral
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from Evaluation import evaluation


def Model_AA_GCN_GRU(Train_Data, Train_Target,sol=None):
    if sol is None:
        sol = [5,5,0.01]
    num_classes = Train_Target.shape[1]
    # Input layers
    X_in = keras.Input(shape=(Train_Data.shape[1],))  # Node features
    A_in = keras.Input(shape=(Train_Data.shape[0],), sparse=True)  # Adjacency matrix

    # GCN layers
    X = spektral.layers.GCNConv(sol[0], activation="relu")([X_in, A_in])
    X = spektral.layers.GCNConv(num_classes, activation="softmax")([X, A_in])

    # Define the model
    model = keras.Model(inputs=[X_in, A_in], outputs=X)

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    layer_no = 4
    model.fit([Train_Data, Train_Data], Train_Target, epochs=sol[1], batch_size=32, verbose=1)
    intermediate_model = models.Model(inputs=model.inputs, outputs=model.layers[layer_no].output)
    Feats = intermediate_model.get_weights()[layer_no]
    Feats = np.resize(Feats, [Train_Data.shape[0], 100])
    learnperc = round(Feats.shape[0] * 0.75)
    X_train = Feats[:learnperc, :]
    y_train = Train_Target[:learnperc, :]
    X_test = Feats[learnperc:, :]
    Y_test = Train_Target[learnperc:, :]
    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(GRU(units=32, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Second GRU layer
    regressorGRU.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Third GRU layer
    regressorGRU.add(GRU(units=128, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Fourth GRU layer
    regressorGRU.add(GRU(units=64, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(units=1))
    # Compiling the RNN
    regressorGRU.compile(optimizer='Adam',learning_rate = sol[2],loss='mean_squared_error') # SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False)
    # Fitting to the training set
    trx_data = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    tex_data = X_test.reshape((X_test.shape[0], X_test[1].shape[0], 1))
    # tey_data = y_train.reshape((y_train.shape[0], y_train.shape[1]))
    pred = np.zeros((Y_test.shape))
    for i in range(y_train.shape[1]): # for all classes
        regressorGRU.fit(trx_data, y_train[:, i], epochs=1, batch_size=150)
        pr = regressorGRU.predict(tex_data).ravel()
        #pr = regressorGRU.predict(tex_data)
        for j in range(pr.shape[0]):
            # pred[j, i] = np.mean(pr[j, :, :])
            pred[j, i] = np.mean(pr[j])
            if np.isnan(pred[j, i]):
                pred[j, i] = 0
    Eval = evaluation(Y_test, pred)
    return Eval
