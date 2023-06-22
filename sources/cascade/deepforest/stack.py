'''
Created on 2 февр. 2023 г.

@author: keen
'''
class Stack:
    def __init__(self):
        self.mem = []
        
    def top(self):
        return self.mem[len(self.mem) - 1]
    
    def push(self, a):
        self.mem.append(a) 
        
    def pop(self):
        self.mem.pop()          
        
