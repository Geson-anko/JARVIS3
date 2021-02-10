

class Parent:
    x:int = 3
#    def __init__(self):
#        pass
    def __call__(self,*args,**kwargs):
        return self.process(*args,**kwargs)

    def PaPro(self):
        print(100)
class Child(Parent):
#    def __init__(self):
#        super(Child,self).__init__()

    def process(self,x1,x2):
        print(x1)
        print(x2)
        print(self.run(x1,x2))
        self.PaPro()

    def run(self,a,b):
        return a+b


p = Child()
p(1,2)
print(p.x)