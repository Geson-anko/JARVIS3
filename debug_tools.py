"""writing utf-8"""

class Debug:
    def __init__(self,log_title):
        self.log_title = log_title

    def log(self,*args):
        print(self.log_title,*args)

if __name__ == '__main__':
    d = Debug('test')
    d.log(1,2,3)