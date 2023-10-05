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

import time
import threading, queue

import pickle

from tfengine import process_evals

import functions

from minimax import MinimaxNode

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
        self.BATCH_SIZE = 64
        # load tf model from disk
        with open('trainedmodel.p', 'rb') as fp:
            model = pickle.load(fp)
            
        # initialize queue for tensorflow predictions
        self.tfq = queue.Queue()
        
        # initialize queue for data to load into search tree
        self.searchq = queue.Queue()
        
        # initialize tensorflow thread and start
        self.tfproc = threading.Thread(target=process_evals, args=(self.tfq,self.searchq, model))
        self.tfproc.start()
        
        # initialize search tree thread and do NOT start yet
        # search thread starts upon first call to search method
        self.searchtreeproc = threading.Thread(target=self.populate_search_tree, args=(self.searchq,))
        
        # initialize the root of the search tree
        self.root = None
        
        # initialize queue for search tree thread
        self.stq = queue.Queue()
        
        # initialize a dict for storage of already evaluated positions
        self.eval_map = {}
        
        self.searching = False
        
        
        self.search_depth = 0
                
        super().__init__(commands, options, stderr, draw_or_resign)
        
    def timer(self, q, n=0):
        
        while(True):
            while(not self.tfq.empty()):
                print(self.tfq.get())
            print(f'----------Time: {n}----------')
            n = n + 1
            time.sleep(1)
            
    def populate_search_tree(self, initboard, q):
        boards_to_eval = [self.root.name[0].fen()]
        nodes_to_update = [self.root]
        
        leaves = [self.root]
        
        current_node = self.root
        
        while self.searching:
            
            for current_node in leaves:
            
                while not self.searchq.empty():        
                    # tf engine pushes results onto searchq
                    # as long as there are results here, there are search nodes that need updating
                    # zip creates a tuple of the object that needs updating, and the value that should be used
                    # list comprehension sets that value to each object for each tuple using setvalue
                    # 
                    
                    [node.setvalue(value, unhide=True) for node, value in zip(nodes_to_update, self.searchq.get().flatten().tolist())]
                    
                    # remove the nodes that were just updated from the list of nodes to update
                    nodes_to_update = nodes_to_update[self.BATCH_SIZE:]
                
                # get the current board
                board = current_node.name[0]
                
                # get a list of moves with checks and captures at the front to optimize
                #   alpha beta search
                movelist = functions.get_checking_moves(board)
                movelist.extend(functions.get_captures(board))
                movelist.extend(functions.get_legal_moves_nonchecking_noncaptures(board))
                
                children = []
                for move in movelist:
                    
                    # create a copy of the board and push the move onto the copy
                    board_copy = board.copy()
                    board_copy.push(move)
                    
                    # see if the position has already been evaluated
                    try:
                        move_eval = self.eval_map[board.fen()]
                        
                        # if so, create a new active minimax node and add it to the list of children
                        # of the current node
                        children.append(MinimaxNode(move_eval, (board_copy, move)))
                        
                    # if the key/eval is missing
                    except KeyError: 
                        
                        # add this position to the list that need to be sent to tensorflow
                        boards_to_eval.append(board_copy.fen())
                        
                        # create a new node that is hidden because it does not have an evaluation yet
                        newnode = MinimaxNode(0, (board_copy, move), hidden = True)
                        
                        # add the new node to the list of nodes that are waiting results from tensorflow
                        nodes_to_update.append(newnode)
                        
                        # add to the current node's list of children
                        children.append(newnode)
                
                # set the minimax node to the list of nodes that were just produced
                current_node.children = children
                
                # increment the depth that has been searched
                self.search_depth = self.search_depth + 1
                print(f'Search depth: {self.search_depth}')
            
            # get the list of nodes at the terminal depth
            leaves = []
            self.root.get_nodes_at_depth(self.search_depth, leaves)
            
            # if there are enough boards for a batch to send to tensorflow
            while len(boards_to_eval) > self.BATCH_SIZE:
                
                # put the first BATCH_SIZE boards on the queue
                self.tfq.put(boards_to_eval[:self.BATCH_SIZE])
                
                # and remove the first BATCH_SIZE elements from the list
                boards_to_eval = boards_to_eval[self.BATCH_SIZE:]
            
            self.root.traversealphabeta(self.search_depth, float('-inf'), float('inf'), self.maxagent)
            
            # should build this into the search tree for speed optimization
            # vvvvv
            # 
            for child in self.root.children:
                if child.value == self.root.value:
                    self.root.name[1] = child.name[1]
                       
        
    def search(self, board: chess.Board, time_limit: chess.engine.Limit, ponder: bool, *args: Any) -> PlayResult:
        
        # set the flag that the engine is searching
        self.searching = True
        
        # if isinstance(time_limit.time, int):
        #     my_time = time_limit.time
        #     my_inc = 0
        # elif board.turn == chess.WHITE:
        #     my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
        #     my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        # else:
        #     my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
        #     my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0
        
        # the engine calls search for the first time once book moves have been exhausted
        if not self.searchtreeproc.is_alive():
            
            self.maxagent = board.turn
            
            self.root = MinimaxNode(0, (board, None), hidden = True)
            self.searchtreeproc.start()
                
        while self.search_depth < 2:
            time.sleep(1)
            print('---sleeping 1s---')
        possibleboards = [] 
        self.root.get_nodes_at_depth(2, possibleboards)
        self.root = possibleboards[possibleboards.index(board)]
        self.search_depth = self.search_depth - 2
            
            
        
        # if ponder is disabled, disable searching right before returning the move to play
        # if not ponder:
        #     self.searching = False
            
        return PlayResult(self.root.name[1], None)