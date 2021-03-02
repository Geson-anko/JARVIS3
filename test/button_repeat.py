import wx

app = wx.App()
frame= wx.Frame(None,title='hello world',size=(300,200))

panel = wx.Panel(frame,wx.ID_ANY)
panel.SetBackgroundColour('#AFAFAF')
layout = wx.BoxSizer(wx.VERTICAL)

for i in range(3):
    button = wx.ToggleButton(panel,wx.ID_ANY,f'トグルボタン{i}',size=(150,50))
    button.SetLabel(f'pushed{i}!')
    layout.Add(button)

panel.SetSizer(layout)

frame.Show()
app.MainLoop()