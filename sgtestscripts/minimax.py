# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:07:37 2023

@author: osiri
"""

import random
    
class MinimaxNode:
    

    def __init__(self, val, parent):
        self.value = val
        self.children = []
        self.parent = parent
    
    def printselfwithparents(self):
        if self.parent==None:
            print(self.getvalue(), end='')

        
        else:
            self.parent.printselfwithparents()
            print('.' + str(self.getvalue()), end='')
        
    
    def printtree(self):  
        self.printselfwithparents()
        if len(self.children) > 0:
            print('\n\tc >> ', end='')
            self.printchildren()
            for child in self.children:
                child.printtree()
        else:
            print()
        
    def printchildren(self):
        if len(self.children) > 0:
            for child in self.children:
                print(child.getvalue(), end="|")
            print()
    
    def isroot(self):
        return self.node.parent == None
    
    def getvalue(self):
        return self.value
    
    def setvalue(self, value):
        self.value = value
    
    def addchildren(self, children):
        self.children.extend(children)
        
    
    def getparent(self):
        return self.parent
    
    def getchildren(self):
        return self.children
    
    def getchild(self, index):
        return self.children[index]
    
    def getmaxvaluechild(self):
        return max([i.getvalue() for i in self.children])
    
    def getminvaluechild(self):
        return min([i.getvalue() for i in self.children])

    def traverse(self, maxagent):
        # print("update call started")
        
        # if len(self.children) == 0: do nothing
        if len(self.children) > 0:
            for child in self.children:
                child.traverse(not maxagent)
            
            if maxagent:
                self.value = self.getmaxvaluechild()
            else:
                self.value = self.getminvaluechild()
        # print("update call finished")
            
    
if __name__ == '__main__':
    random.seed(42)
    node = MinimaxNode(1, None)
    newchildren = []
    for i in range(0,10):
        newchildren.append(MinimaxNode(random.randint(-10,10), node))
        
    node.addchildren(newchildren)
    
    for x in range(len(node.getchildren())):
        
        newchildren2 = []
        for i in range(0,4):
            newchildren2.append(MinimaxNode(random.randint(-10,10), newchildren[0]))
        newchildren[x].addchildren(newchildren2)
    
    

    node.printtree()
    
    node.traverse(True)
    
    node.printtree()
    