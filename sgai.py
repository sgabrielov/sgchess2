#!/usr/local/bin/python3

import strategies
import random

class sgai2(MinimalEngine):
    def search(self, board, *args):
        return PlayResult(random.choice(list(board.legal_moves)), None)

class sgai(ExampleEngine):
    """Gets the first move when sorted by uci representation"""
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)
