# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:07:37 2023

@author: osiri
"""

import random, time
    
class MinimaxNode:
    

    def __init__(self, val, parent, hidden=True):
        self.value = val
        self.children = []
        self.parent = parent
        self.hidden = hidden
        
    def setvalue(self, val, unhide=True):
        self.value = val
        if unhide:
            self.hidden = False
            
    def printselfwithparents(self):
        if self.parent==None:
            print(self.getvalue(), end='')

        
        else:
            self.parent.printselfwithparents()
            print('.' + str(self.getvalue()), end='')
        
    
    def printtree(self):
        if self.hidden:
            return
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
    
    def addchildren(self, children):
        self.children.extend(children)
    
    def isleaf(self):
        return len(self.children) == 0
    
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

    # recursive minimax traversal
    def traverse(self, maxagent):
        # print("update call started")
        
        # if len(self.children) == 0: do nothing
        if len(self.children) > 0 and not self.hidden:
            for child in self.children:
                child.traverse(not maxagent)
            
            if maxagent:
                self.value = self.getmaxvaluechild()
            else:
                self.value = self.getminvaluechild()
        # print("update call finished")
    
    # recursive minimax with a/b pruning
    def traversealphabeta(self, depth, a, b, maxagent):
        if depth == 0 or self.isleaf() or self.hidden:
            return self.value
        
        if maxagent:
            value = float('-inf')
            for child in self.getchildren():
                value = max(value, child.traversealphabeta(depth-1, a, b, not maxagent))
                if value > b or child.hidden:
                    #print(f'pruned {value}')
                    break
                a = max(a, value)
            self.value = value
            return value
                
        else:
            value = float('inf')
            for child in self.getchildren():
                value = min(value, child.traversealphabeta(depth-1, a, b, not maxagent))
                if value < a or child.hidden:
                    #print(f'pruned {value}')
                    break
                b = min(b, value)
            self.value = value
            return value
        
    def get_nodes_at_depth(self, depth, outlist = []):
        
        if depth == 1:
            outlist.extend(self.children)
        elif depth > 1:
            for child in self.children:
                child.get_nodes_at_depth(depth - 1, outlist)
    
    def __eq__(self,o):
        return self.value == o

def testfunc(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    print(end - start)



if __name__ == '__main__':
    random.seed(42)
    node = MinimaxNode(1, None, hidden=False)
    newchildren = []
    for i in range(0,10):
        newchildren.append(MinimaxNode(random.randint(-10,10), node, hidden=False))
        
    node.addchildren(newchildren)
    
    for x in range(len(node.getchildren())):
        
        newchildren2 = []
        for i in range(0,4):
            newchildren2.append(MinimaxNode(random.randint(-10,10), newchildren[0], hidden=False))
        newchildren[x].addchildren(newchildren2)
    #childlist = []
    #node.get_nodes_at_depth(2, childlist)
    #print([child.value for child in childlist])
    

    node.printtree()
    
    testfunc(node.traversealphabeta, *(10, float('-inf'), float('inf'), True,))
    #testfunc(node.traverse, *(True,))
    
    node.printtree()
    