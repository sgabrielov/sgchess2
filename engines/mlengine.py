# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:20:36 2023

@author: osiri
"""

from strategies import ExampleEngine
from typing import Any
import chess
import random
from chess.engine import PlayResult

class RandomMove2(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)