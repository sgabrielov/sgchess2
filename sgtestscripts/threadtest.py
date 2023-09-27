# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:48:03 2023

@author: osiri
"""

# for testing tensorflow memory issues using threads

import threading

import tensorflow as tf
import pickle

import numpy as np

import sys

import chess

import time

starting_fen = '8/1p6/p3p3/1pPk4/1P2pPp1/8/4K3/6b1 w - - 0 46'

SCRIPTLOCATION = '/home/ml/sgchess/'
if SCRIPTLOCATION not in sys.path:
    sys.path.append(SCRIPTLOCATION)

BATCH_SIZE = 64

eval_map = {}
max_depth = 1

TREE_ROOT = chess.Board(starting_fen)

class MinimaxNode():
    
    
    def __init__(self, init_board = starting_fen, maxagent = True):
        
        self.board = chess.Board(starting_fen)
        self.children = self.board.legal_moves()
    
def get_eval(board, model):
    try:
        return eval_map[board.fen()]
    except KeyError:
        eval_estimate = model.predict(convert_fen_to_bitboard(board.fen())[None])[0][0]
        eval_map[board.fen()] = eval_estimate
        return eval_estimate
    
def get_batched_eval(board, model, batch_size = BATCH_SIZE):
    
    pass

def populate_search_tree(initial_position: chess.Board):
    
    with open(SCRIPTLOCATION + 'trainedmodel.p', 'rb') as fp:
        model = pickle.load(fp)
        
    print(get_eval(initial_position, model))

def get_checking_moves(board):
    return [move for move in board.legal_moves if board.gives_check(move)]

def get_legal_moves(board):
    return [move for move in board.legal_moves]

def get_captures(board):
    return [move for move in board.legal_moves if board.is_capture(move)]

def get_legal_moves_nonchecking_noncaptures(board):
    return [move for move in board.legal_moves if not board.gives_check(move) and not board.is_capture(move)]

def testfunc(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    print(end - start)
    
def convert_fen_to_bitboard(fen, cols=None) -> np.ndarray:
    
    
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
    
    board = chess.Board(fen)
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

    return tf.convert_to_tensor(outlist)   

if __name__ == '__main__':
    t1 = threading.Thread(target=populate_search_tree, args=(chess.Board(starting_fen),))
    t1.start()
    t1.join()
    
    print("Done")