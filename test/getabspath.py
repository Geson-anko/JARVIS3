import os
def getabspath(file_name):
    dirs = os.path.dirname(os.path.abspath(__file__)).split('\\')
    dirs = '/'.join(dirs) +'/'+ file_name
    return dirs

print(getabspath('aaa.txt'))
print(__file__)