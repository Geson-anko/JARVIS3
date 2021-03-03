import wx

class MyFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, "Test", size=(200,150))
        Pan = wx.ScrolledWindow(self, -1)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        StTxt1 = wx.StaticText(Pan, -1, "AAAAAA")
        StTxt2 = wx.StaticText(Pan, -1, "BBBBBB")
        StTxt3 = wx.StaticText(Pan, -1, "CCCCCC")
        StTxt4 = wx.StaticText(Pan, -1, "DDDDDD")
        StTxt5 = wx.StaticText(Pan, -1, "EEEEEE")
        StTxt6 = wx.StaticText(Pan, -1, "FFFFFF")
        StTxt7 = wx.StaticText(Pan, -1, "GGGGGG")
        
        sizer.Add(StTxt1, 0, wx.ALIGN_LEFT|wx.ALL, 5)
        sizer.Add(StTxt2, 0, wx.ALIGN_LEFT|wx.ALL, 5)
        sizer.Add(StTxt3, 0, wx.ALIGN_LEFT|wx.ALL, 5)
        sizer.Add(StTxt4, 0, wx.ALIGN_LEFT|wx.ALL, 5)
        sizer.Add(StTxt5, 0, wx.ALIGN_LEFT|wx.ALL, 5)
        sizer.Add(StTxt6, 0, wx.ALIGN_LEFT|wx.ALL, 5)
        sizer.Add(StTxt7, 0, wx.ALIGN_LEFT|wx.ALL, 5)
        Pan.SetSizer(sizer)
        Pan.SetScrollRate(10,10)
        self.Show()

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame()
        return True

app = MyApp()
app.MainLoop()

class Toggles(wx.Frame):
    def __init__(self):
        super().__init__(self,None,wx.ID_ANY,"Toggles",size=(500,500))
        Pan = wx.ScrolledWindow(self,wx.ID_ANY)

        sizer = wx.BoxSizer(wx.VERTICAL)

        for i in range(4):
            b = wx.ToggleButton(Pan,wx.ID_ANY,f'toggle{i}')
            sizer.Add(b)
        Pan.SetSizer(sizer)
        Pan.SetScrollRate(10,10)
        self.Show()

class MyApp(wx.App):
    def OnInit(self):
        self.frame = Toggles()
        return True

app = MyApp()
app.MainLoop()