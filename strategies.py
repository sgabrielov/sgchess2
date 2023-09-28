"""
Some example strategies for people who want to create a custom, homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""

from __future__ import annotations
import chess
from chess.engine import PlayResult
import random
from engine_wrapper import MinimalEngine
from typing import Any, Union
import logging

import threading
import time

import tensorflow as tf
import pickle
import pandas as pd


MOVE = Union[chess.engine.PlayResult, list[chess.Move]]


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    pass


# Strategy names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self, board: chess.Board, time_limit: chess.engine.Limit, ponder: bool, draw_offered: bool,
               root_moves: MOVE) -> chess.engine.PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)

class TFEngine(ExampleEngine):
    
    def __init__(self, commands, options, stderr, draw_or_resign, **popen_args: str) -> None:
        
        with open("trainedmodel.p", "rb") as fp:
            self.model = pickle.load(fp)
        
        #self.t1 = threading.Thread(target=self.timer, args=(0,))
        #self.t1.start()
        
        self.eval_map = {}
        super().__init__(commands, options, stderr, draw_or_resign)
    
    def timer(self, n=0):
        
        while(True):
            print(f'----------Time: {n}----------')
            n = n + 1
            time.sleep(1)
            
    def populate_search_tree(self, model: tf.keras.src.engine.sequential.Sequential):
        
        pass
    
    def convert_fen_to_bitboard(self, board: chess.Board, cols=None) -> pd.core.series.Series:
        
        
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
    
        return tf.convert_to_tensor(outlist)
    
    def search(self, board: chess.Board, ponder: bool, *args: Any) -> PlayResult:
        
        
        # print the estimated evaluation of the current board
        # print(self.model.predict(self.convert_fen_to_bitboard(board.fen()).values[None]))
        
        
        # for now, return a random move
        maxagent = board.turn
        if maxagent:
            topeval = -1000
        else:
            topeval = 1000
            
        for move in board.legal_moves:
            boardcopy = board.copy()
            boardcopy.push(move)
            
            ev = self.get_eval(boardcopy)
            if maxagent:
                if ev > topeval:
                    topmove = move
            else:
                if ev < topeval:
                    topmove = move
            
        return PlayResult(topmove, None)
    
    def get_eval(self, board):
        try:
            return self.eval_map[board.fen()]
        except KeyError:
            eval_estimate = self.model.predict(self.convert_fen_to_bitboard(board)[None])[0][0]
            self.eval_map[board.fen()] = eval_estimate
            return eval_estimate

class minimax_node():
    
    def __init__(self):
        pass