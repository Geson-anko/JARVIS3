import tkinter as tk
from tkinter import Scrollbar, messagebox
import multiprocessing as mp
from typing import Tuple
from .config import config

class SwitchPanel(tk.Frame):
    def __init__(self,master,switch_obj:Tuple[str,mp.Value]) -> None:
        """
        master: tkinter object
        switch_obj: (title, switch)
        """
        super().__init__(master)
        self.master = master
        self.title = switch_obj[0]
        self.switch = switch_obj[1]
        
        self.label = tk.Label(master,text=self.title,font=config.switch_font)
        self.label.pack(side=tk.TOP)
        self.on_img = tk.PhotoImage(file=config.on_image)
        self.off_img = tk.PhotoImage(file=config.off_image)
        img = self.on_img if self.switch.value else self.off_img
        self.button = tk.Button(master,image=img,command=self.switching)
        self.button.pack(side=tk.TOP)
    
    def switching(self) -> None:
        if self.switch.value:
            self.button.config(image=self.off_img)
            self.switch.value = False
        else:
            self.button.config(image=self.on_img)
            self.switch.value = True


class SwitchScroller(tk.Frame):
    def __init__(self,master,switch_objects:Tuple[Tuple[str,mp.Value],...]) -> None:
        """
        master: tkinter object
        switch_objects: ((title, switch),...)
        """
        super().__init__(master)
        self.master = master
        self.label = tk.Label(master,text=config.scroll_title,font=config.scroll_font)
        self.label.pack(side=tk.TOP)
        canvas = tk.Canvas(master)
        bar = tk.Scrollbar(master,orient=tk.VERTICAL)
        bar.pack(side=tk.RIGHT,fill=tk.BOTH)
        bar.config(command=canvas.yview)

        canvas.config(yscrollcommand=bar.set)
        canvas.config(scrollregion=(0,0,0,int(len(switch_objects) * 70)))
        canvas.pack(side=tk.TOP)

        frame = tk.Frame(canvas)
        canvas.create_window((0,0),window=frame,anchor=tk.NW,width=canvas.cget('width'))
        for switch in switch_objects:
            SwitchPanel(frame,switch)


class ShutdownButton(tk.Frame):
    def __init__(self,master,shutdown_switch:mp.Value):
        """
        master: tkinter object
        shutdown_switch:multiprocessing shared memory bool value.
        """
        super().__init__(master)
        self.master = master
        self.switch = tk.Button(master,text=config.shutdown_title,command=self.shutdown_check,font=config.shutdown_font)
        self.switch.pack(side=tk.TOP)
        self.shutdown = shutdown_switch

    def shutdown_check(self):
        self.shutdown.value = messagebox.askyesno(config.shutdown_title,config.shutdown_ask)
        if self.shutdown.value:
            self.master.destroy()
    
        

class GUIController(tk.Tk):
    def __init__(self,shutdown:mp.Value,switch_objects:Tuple[Tuple[str,mp.Value],...]) -> None:
        """
        shutdown_switch:multiprocessing shared memory bool value.
        switch_objects: ((title, switch),...)
        """
        super().__init__()
        self.title(config.controler_title)
        self.geometry(config.root_geometry)
        ShutdownButton(self,shutdown)
        SwitchScroller(self,switch_objects)

    def __call__(self):
        self.mainloop()


        
if __name__ == '__main__':
    
    shutdown = mp.Value('i',False)
    num = 10
    switches = [mp.Value('i',True) for _ in range(num)]
    titles = [f'switch {i}' for i in range(num)]
    switch_objs = list(zip(titles,switches))
    cotroler = GUIControler(shutdown,switch_objs)
    cotroler()
    
    #root = tk.Tk()
    #root.geometry('300x300')
    #SwitchPanel(root,('test',mp.Value('i')))
    #root.mainloop()
    
    #root = tk.Tk()
    #root.geometry('300x300')
    #SwitchScroller(root,(('test',mp.Value('i')),))
    #root.mainloop()