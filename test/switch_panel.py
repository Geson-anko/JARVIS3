import tkinter as tk
from typing import Sized

class SwitchPanel(tk.Frame):
    is_on:bool = False
    def __init__(self,master,conf='top'):
        super().__init__(master)
        self.master = master
        self.lt = tk.Label(master,text='Switch',font=('Helvetica',10)).pack(side=conf)
        self.on_img = tk.PhotoImage(file='on.png')
        self.off_img = tk.PhotoImage(file='off.png')
        img = self.on_img if self.is_on else self.off_img
        self.on_button = tk.Button(master,image=img,command=self.switch)
        self.on_button.pack(side=conf)
        self.conf = conf

    def switch(self):
        if self.is_on:
            self.on_button.config(image=self.off_img)
            self.is_on =False
        else:
            self.on_button.config(image=self.on_img)
            self.is_on = True

        print(self.conf,self.is_on)

class scroller(tk.Frame):
    def __init__(self,master,switches:int=1):
        super().__init__(master)
        self.master = master
        tk.Label(master,text='Scroll',font=('Helvetica',15)).pack(side=tk.TOP)
        canvas = tk.Canvas(master)

        bar_y = tk.Scrollbar(master,orient=tk.VERTICAL)
        bar_y.pack(side=tk.RIGHT,fill=tk.BOTH)
        bar_y.config(command=canvas.yview)

        canvas.config(yscrollcommand=bar_y.set)
        canvas.config(scrollregion=(0,0,0,switches*70))
        canvas.pack(side=tk.TOP)
 #       listbox = tk.Listbox(canvas,yscrollcommand=bar_y.set)
        frame = tk.Frame(canvas)
        canvas.create_window((-40,0),window=frame,anchor=tk.NW,width=canvas.cget('width'))

        for i in range(switches):
            SwitchPanel(frame)
#        listbox.pack(side=tk.TOP)
        



root = tk.Tk()
root.geometry('300x300')
#app = SwitchPanel(root)
#app2 = SwitchPanel(root,'top')
scroller(root)
root.mainloop()
