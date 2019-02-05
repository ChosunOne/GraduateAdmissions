from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

path = "Admission_Predict.csv"
df = pd.read_csv(path)

cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'Chance of Admit ']
sns.pairplot(df[cols], diag_kind="kde")
plt.show()

X = pd.DataFrame()

good_cols = ['GRE Score', 'TOEFL Score', 'CGPA', 'SOP', 'LOR ', 'Research']
sc = StandardScaler()

for cat in good_cols:
    X[cat] = df[cat].values

y = pd.DataFrame()

y['Chance of Admit '] = df['Chance of Admit '].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

norm_X_train = sc.fit_transform(X_train)
norm_X_test = sc.fit_transform(X_test)

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    return model


model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.summary()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 1000

history = model.fit(norm_X_train, y_train, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Chance of Admit]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 0.5])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$Chance of Admit^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 0.5])
    plt.show()


plot_history(history)

loss, mae, mse = model.evaluate(norm_X_test, y_test, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} Chance of Admit".format(mae))

test_predictions = model.predict(norm_X_test).flatten()

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [Chance of Admit]')
plt.ylabel('Predictions [Chance of Admit]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

plt.show()


