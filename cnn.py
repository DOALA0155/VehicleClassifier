from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from image_processing import get_image_data
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

images, labels = get_image_data()
x_train, x_test, y_train, y_test = train_test_split(images, labels)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)
model = Sequential()

model.add(Conv2D(120, kernel_size=(3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(90, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(100, activation="relu"))
model.add(Dense(5, activation="softmax"))
print(model.summary())

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
train_model = model.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=10)

score_history = train_model.history
print(score_history)
acc_history = score_history["accuracy"]
val_acc_history = score_history["val_accuracy"]
loss_history = score_history["loss"]
val_loss_history = score_history["val_loss"]
x = range(len(acc_history))

plt.plot(x, acc_history, label="train_accuracy")
plt.plot(x, val_acc_history, label="test_accuracy")
plt.legend(loc="best")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.plot(x, loss_history, label="train_loss")
plt.plot(x, val_loss_history, label="test_loss")
plt.legend(loc="best")
print("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

score = model.evaluate(x_test, y_test)
print("Test loss: {}".format(score[0]))
print("Score: {}".format(score[1]))
model.save("./Models/CnnModel3.h5")
