# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:48:03 2023

@author: osiri
"""

# for testing tensorflow memory issues using threads

import threading, queue

import pickle
import tensorflow as tf

import sys

import chess

import time


import random

starting_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
sample_fen = '8/1p6/p3p3/1pPk4/1P2pPp1/8/4K3/6b1 w - - 0 46'

SCRIPTLOCATION = '/home/ml/sgchess/'
if SCRIPTLOCATION not in sys.path:
    sys.path.append(SCRIPTLOCATION)

BATCH_SIZE = 64

tfexit = False

eval_map = {}
max_depth = 1

TREE_ROOT = chess.Board(starting_fen)

class MyClass():
    
    def __init__(self, val):
        self.val = val
    
    def setval(self, newval):
        self.val = newval
        
    def __str__(self):
        return str(self.val)
    
    
def process_evals(inq: queue.Queue, tfmodel):
    
    while True:
        
        boards = inq.get()
        out = get_batched_eval(boards, tfmodel)
        print(out)
        print(type(out))
        print(out.shape)
        inq.task_done()
        
def get_batched_eval(input_data: list, model, batch_size = 64):
    
    # input is a list of chess.Board boards of length BATCH_SIZE
    # apply the bitboard conversion to every board using map and sture the result in a list
    # convert the list to a tf tensor and send to predict method
    # result is a BATCH_SIZE length numpy array containing predictions
    
    # return model.predict(tf.convert_to_tensor(list(map(convert_fen_to_bitboard, boards))))
    return model.predict(tf.convert_to_tensor(input_data))

def get_checking_moves(board):
    return [move for move in board.legal_moves if board.gives_check(move)]

def get_legal_moves(board):
    return [move for move in board.legal_moves]

def get_captures(board):
    return [move for move in board.legal_moves if board.is_capture(move)]

def get_legal_moves_nonchecking_noncaptures(board):
    return [move for move in board.legal_moves if not board.gives_check(move) and not board.is_capture(move)]

def timefunc(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    print(f'{func.__name__}: {end - start}s')
    
def convert_fen_to_bitboard(board: chess.Board, cols=None):
    
    
    """Converts a fen string to a bitboard mapping
        
        Parameters
        ----------
        fen : str
            The FEN string
            
        Returns
        -------
        list
            A list of bool
    """
    
    # The bitboard mapping is going to use 1 hot encoding - where each bit
    # corresponds to a specific square, piece, and color
    
    outlist = []
    
    # encode white pieces
    # in python-chess chess.WHITE = True and chess.BLACK = False
    # chess.Pawn = 1, King = 6, etc
    for i in range(1,7):
        outlist.extend(board.pieces(i, chess.WHITE).tolist())
    
    # encode castling rights for white
    
    outlist.append(board.has_castling_rights(chess.WHITE))
    outlist.append(board.has_queenside_castling_rights(chess.WHITE))
    
    # encode black pieces
    for i in range(1,7):
        outlist.extend(board.pieces(i, chess.BLACK).tolist())
    
    # encode castling rights for black
    
    outlist.append(board.has_castling_rights(chess.BLACK))
    outlist.append(board.has_queenside_castling_rights(chess.BLACK))
    
    return outlist

def main():
    with open('trainedmodel.p', 'rb') as fp:
        model = pickle.load(fp)
    #t1 = threading.Thread(target=populate_search_tree, args=(chess.Board('2r4k/p4b2/4pq2/1p1p1nR1/5P2/P2B4/1P2Q2P/1K4R1 b - - 2 30'),))
    #t1.start()
    #t1.join()
    
    # generate batch_size identical boards
    boards = [chess.Board('2r4k/p4b2/4pq2/1p1p1nR1/5P2/P2B4/1P2Q2P/1K4R1 b - - 2 30') for x in range(BATCH_SIZE)]
    # play the first legal move on the first board, to simulate variety
    boards[0].push(list(boards[0].legal_moves)[0])
    
    boards2 = [chess.Board(boards[0].fen()) for x in range(BATCH_SIZE)]
    boards2[0].push(list(boards2[0].legal_moves)[0])
    
    
        
    #print(get_batched_eval(boards, model))
    
    # timefunc(get_batched_eval, *(boards, model,))
    
    
    q1 = queue.Queue()
    q1.put(boards)
    q1.put(boards2)
    t1 = threading.Thread(target=process_evals, args=(model, q1,))
    t1.start()
    print('work started')
    q1.join()
    print('work finished')
    
    
    
    
    # n = SimpleMinimaxNode(5)
    # l = [SimpleMinimaxNode(random.randint(-10,10)) for x in range(10)]
    # n.set_children(l)
    # n.print_tree()

    
    print("Done")

def main2():
    
    random.seed(42)
    ls1 = [MyClass(0) for _ in range(100)]
    ls2 = [random.randint(-10,10) for _ in range(100)]
    
    print(ls2)
    
    print([node.val for node in ls1])
    
    [node.setval(value) for node, value in zip(ls1, ls2)]
    
    print([node.val for node in ls1])
    
    
if __name__ == '__main__':
    main2()
