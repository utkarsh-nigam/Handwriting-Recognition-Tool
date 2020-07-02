import tkinter as tk
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pickle
import os,sys
import math

array=np.zeros((200,800),int)
new_array=[]
character_list=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.title("Hand Writing Translator")
        self.rowconfigure(0, minsize=200, weight=1)
        self.rowconfigure(1, minsize=50, weight=1)
        self.columnconfigure(0, minsize=800, weight=1)

        self.canvas = tk.Canvas(self, width=800, height=200, bg = "white", cursor="circle",insertwidth=5,borderwidth=2,relief="solid")
        self.canvas.grid(row=0, column=0, padx=(10,10), pady=10,sticky="w")

        self.fr_buttons = tk.Frame(self)
        self.fr_buttons.columnconfigure(0, minsize=550, weight=0)
        self.fr_buttons.columnconfigure(1, minsize=100, weight=0)
        self.label = tk.Label(self.fr_buttons, width=42, height=3, background="white",borderwidth=2,relief="solid", anchor="w", justify="left")
        self.label['text'] = " Predicted Value: "
        self.label.config(font=("Courier", 24))
        self.label.grid(row=0, column=0, padx=(3,0), sticky="w")
        self.fr_buttons2 = tk.Frame(self.fr_buttons)
        self.fr_buttons2.rowconfigure(0, minsize=10, weight=0)
        self.fr_buttons2.rowconfigure(1, minsize=10, weight=0)
        self.button_clear = tk.Button(self.fr_buttons2,text = "Clear", width=17, height=2, command = self.clear_all)
        self.button_clear.grid(row=0, column=0, padx=(10,0), pady=4,sticky="e")
        self.button_submit = tk.Button(self.fr_buttons2, text="Translate",width=17,height=2, command=self.submit)
        self.button_submit.grid(row=1, column=0, padx=(10,0), pady=4, sticky="e")
        self.fr_buttons2.grid(row=0, column=1, padx=(0,0),sticky="e")
        self.fr_buttons.grid(row=1, column=0, padx=(10,10), pady=10,sticky="w")

        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)
        filename = 'MLP_Adam_100_AllData_sigmoid_earlystop.sav'
        self.loaded_model = pickle.load(open(filename, 'rb'))

    def clear_all(self):
        self.points_recorded = []
        self.canvas.delete("all")
        self.label['text'] = " Predicted Value: "

    def submit(self):
        self.my_array=self.points_recorded
        self.write_dataframe()

    def print_points(self):
        if self.points_recorded:
            self.points_recorded.pop()
            self.points_recorded.pop()
        self.canvas.create_oval(self.points_recorded, fill = "black")
        self.points_recorded[:] = []

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    def draw_from_where_you_are(self, event):
        if self.points_recorded:
            self.points_recorded.pop()
            self.points_recorded.pop()

        self.x = event.x
        self.y = event.y
        self.canvas.create_oval(self.previous_x, self.previous_y,
                                self.x, self.y,fill="black",width=5)
        self.points_recorded.append(self.previous_x)
        self.points_recorded.append(self.previous_y)
        self.points_recorded.append(self.x)
        self.points_recorded.append(self.x)
        self.previous_x = self.x
        self.previous_y = self.y

    def write_dataframe(self):
        self.final_write=pd.DataFrame()
        self.array = np.full((200, 800), 0)

        #try:
        for i in range(0, int((len(self.my_array)) / 2)):
            cord_1, cord_2 = self.my_array[(2 * i) + 1], self.my_array[2 * i],
            if (cord_1<195 and cord_1>5 and cord_2<795 and cord_2>5):
                for k in range(0, 10):
                    for j in range(0, 10):
                        self.array[cord_1 + 5 - k,cord_2 + 5 - j,] = 255

        self.check = np.arange(0, 800).tolist()
        self.final_check = set(self.check) - set(np.where(~self.array.any(axis=0))[0].tolist())
        self.final_check = list(self.final_check)  # .tolist()

        self.start_array = []
        self.range_array = []

        for k in range(0, len(self.final_check) - 1):
            range_letter = self.final_check[k + 1] - self.final_check[k]
            if (range_letter > 9):
                self.range_array.append(range_letter)

        self.word_flag = 0
        self.pixel_count = 0

        for i in range(0, len(self.final_check) - 1):
            range_letter = self.final_check[i + 1] - self.final_check[i]
            if (range_letter == self.range_array[self.word_flag]):
                self.start_array.append(self.final_check[i - self.pixel_count])
                self.start_array.append(self.final_check[i])
                self.pixel_count = 0
                if (self.word_flag == len(self.range_array) - 1):
                    continue
                else:
                    self.word_flag += 1
            else:
                self.pixel_count += 1

        if (self.pixel_count > 1):
            self.start_array.append(self.final_check[-1 - self.pixel_count])
            self.start_array.append(self.final_check[-1])

        for j in range(0, int(len(self.start_array) / 2)):
            new_np = self.array[:, self.start_array[(2 * j)]:self.start_array[(2 * j) + 1]]
            row_check = np.where(~new_np.any(axis=1))[0].tolist()

            for a in range(0, len(row_check) - 1):
                row_difference = row_check[a + 1] - row_check[a]
                if (row_difference > 9):
                    start_row = row_check[a] + 1
                    end_row = row_check[a + 1] - 1
                    break
            new_np = new_np[start_row:end_row, :]
            pad_h = 200 - new_np.shape[0]
            pad_v = 200 - new_np.shape[1]
            if (pad_h % 2 == 0):
                pad_h1, pad_h2 = pad_h / 2, pad_h / 2
            else:
                pad_h1, pad_h2 = math.floor(pad_h / 2), math.ceil(pad_h / 2)

            if (pad_v % 2 == 0):
                pad_v1, pad_v2 = pad_v / 2, pad_v / 2
            else:
                pad_v1, pad_v2 = math.floor(pad_v / 2), math.ceil(pad_v / 2)
            new_np = np.pad(new_np, ((int(pad_h1), int(pad_h2)), (int(pad_v1), int(pad_v2))), 'constant',
                            constant_values=(0, 0))
            plt.imshow(new_np, cmap="gray", aspect='equal')
            plt.axis('off')
            plt.savefig('Letter' + str(j + 1) + '.png', bbox_inches='tight')
            image = Image.open('Letter' + str(j + 1) + '.png').convert('L')
            os.remove('Letter' + str(j + 1) + '.png')
            image = image.resize((28, 28), Image.ANTIALIAS)
            data = np.asarray(image)

            data = np.array([data.ravel()])
            temp_df = pd.DataFrame(data=data)
            self.final_write = self.final_write.append(temp_df)
        self.predict_word()
        #except:
        #    self.points_recorded = []
        #    self.canvas.delete("all")
        #    self.label['text'] = " Didnt Get The Word! Please Redraw!"

    def predict_word(self):
        global character_list
        y_pred = self.loaded_model.predict(self.final_write)
        output_word =" Predicted Value: "
        for char_index in y_pred:
            output_word=output_word+str(character_list[char_index])
        self.label['text'] = output_word


if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()