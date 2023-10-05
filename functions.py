# -*- coding: utf-8 -*-

import chess, time

def convert_fen_to_bitboard(board: chess.Board, cols=None) -> list:
    
    
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

def testfunc(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    print(end - start)
    
def get_checking_moves(board):
    return [move for move in board.legal_moves if board.gives_check(move)]

def get_legal_moves(board):
    return [move for move in board.legal_moves]

def get_captures(board):
    return [move for move in board.legal_moves if board.is_capture(move) and not board.gives_check(move)]

def get_legal_moves_nonchecking_noncaptures(board):
    return [move for move in board.legal_moves if not board.gives_check(move) and not board.is_capture(move)]

