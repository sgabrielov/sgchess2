# -*- coding: utf-8 -*-

class MinimaxNode:
    def __init__(self, val, name:tuple, hidden = False):
        self.value = val
        self.children = []
        
        # name identifies the node as a tuple containing objects
        self.name = name
        
        self.hidden = hidden
    
    def setvalue(self, val, unhide=True):
        self.value = val
        if unhide:
            self.hidden = False
            
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
        
    def isleaf(self):
        return len(self.children) == 0
    
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
    
    def __eq__(self, o):
        return self.name[0] == o
    
    def __del__(self):
        for child in self.children:
            del child
            
        del self
        

    
            
# if __name__ == '__main__':
    
