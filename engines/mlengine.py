# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:11:28 2023

@author: osiri
"""

from strategies import ExampleEngine
import random
import chess
from chess.engine import PlayResult
from typing import Any

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)