import tkinter as tk

class Switching:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('On/Off Switch')
        self.root.geometry("500x300")
        self.is_on = True

        self.my_label = tk.Label(
            self.root,
            text="The Switch Is ON",
            fg='green',
            font=("Helvetica",32)
        )
        self.my_label.pack(pady=30)
        self.on__img = tk.PhotoImage(file='on.png')
        self.off_img = tk.PhotoImage(file='off.png')
        self.on_button = tk.Button(self.root,image=self.on__img,bd=0,command=self.switch)
        self.on_button.pack(pady=50)

    def switch(self):
        if self.is_on:
            self.on_button.config(image=self.off_img)
            self.my_label.config(text='The Switch is Off',fg='grey')
            self.is_on =False
        else:
            self.on_button.config(image=self.on__img)
            self.my_label.config(text='The Switch is On',fg="green")
            self.is_on = True

    def __call__(self):
        self.root.mainloop()

if __name__ == '__main__':
    gui = Switching()
    gui()




