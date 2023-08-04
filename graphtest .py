import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import *
from PIL import ImageTk, Image

data = pd.read_csv("/Users/shreyas/Desktop/TSLA.csv")
print('Raw data from tesla dataset : ')
print(pd.options.display.max_rows)


# print(data.head())


def import_csv_data():
    global v
    csv_file_path = askopenfilename()
    print(csv_file_path)
    v.set(csv_file_path)
    df = pd.read_csv(csv_file_path)
    print(df.head())
    # data = pd.read_csv("/Users/shreyas/Desktop/TSLA.csv")


root = tk.Tk()
tk.Label(root, text='File Path').grid(row=0, column=0)
v = tk.StringVar()
entry = tk.Entry(root, textvariable=v).grid(row=0, column=1)
tk.Button(root, text='Browse Data Set', command=import_csv_data).grid(row=1, column=0)
tk.Button(root, text='Close', command=root.destroy).grid(row=1, column=1)
root.mainloop()

data = data.drop('Date', axis=1)
data = data.drop('Adj Close', axis=1)
print('\n\nData after removing Date and Adj Close : ')
print(data.head())

data_X = data.loc[:, data.columns != 'Close']
data_Y = data['Close']
train_X, test_X, train_y, test_y = train_test_split(data_X, data_Y, test_size=0.25)
print('\n\nTraining Set')
print(train_X.head())
print(train_y.head())

# calling a variable regressor
regressor = LinearRegression()
regressor.fit(train_X, train_y)

predict_y = regressor.predict(test_X)
score = regressor.score(test_X, test_y)
print('Prediction Score : ', score)
mse = mean_squared_error(test_y, predict_y)
print('Mean Squared Error : ', mse)

s2 = "Prediction Score:" + str(score)
m2 = "MSE:" + str(mse)

root2 = Tk()


def click():
    myllabel = Label(root2, text=s2, fg="blue", bg="yellow")
    myllabel.pack()
    myllabel2 = Label(root2, text=m2, fg="red", bg="yellow")
    myllabel2.pack()


myButton = Button(root2, text="values", command=click)
myButton.pack()
root2.mainloop()

root = Tk()
root.title('tesla stock prediction')
root.geometry("400x200")


def graph():
    fig = plt.figure()
    ax = plt.axes()
    ax.grid()
    ax.set(xlabel='Close ($)', ylabel='Open ($)', title='Tesla Stock Prediction using Linear Regression')
    ax.plot(test_X['Open'], test_y)
    ax.plot(test_X['Open'], predict_y)
    fig.savefig('LRPlot.png')
    plt.show()


my_button = Button(root, text="PRESS", padx=30, command=graph, fg="red")
root.configure(bg='blue')
my_button.pack()

root.mainloop()
