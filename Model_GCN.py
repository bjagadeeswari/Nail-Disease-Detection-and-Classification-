import keras
import numpy as np
import spektral
from Evaluation import evaluation


def Model_GCN(Train_Data, Train_Target, Test_Data, Test_Target):
    num_classes = Train_Target.shape[1]
    # Input layers
    X_in = keras.Input(shape=(Train_Data.shape[1],))  # Node features
    A_in = keras.Input(shape=(Train_Data.shape[0],), sparse=True)  # Adjacency matrix

    # GCN layers
    X = spektral.layers.GCNConv(16, activation="relu")([X_in, A_in])
    X = spektral.layers.GCNConv(num_classes, activation="softmax")([X, A_in])

    # Define the model
    model = keras.Model(inputs=[X_in, A_in], outputs=X)

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the model
    model.fit([Train_Data, Train_Data], Train_Target, epochs=50, batch_size=32, verbose=1)
    predict = model.predict(Test_Data).astype('int')
    Eval = evaluation(predict, Test_Target)
    return np.asarray(Eval).ravel()

