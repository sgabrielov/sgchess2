# -*- coding: utf-8 -*-

import tensorflow as tf
import queue

def process_evals(inq: queue.Queue, outq: queue.Queue, tfmodel):
    
    while True:
        
        boards = inq.get()
        outq.put(get_batched_eval(boards, tfmodel))
        inq.task_done()
        
def get_batched_eval(input_data: list, model, batch_size = 64):
    
    # input is a list of chess.Board boards of length BATCH_SIZE
    # apply the bitboard conversion to every board using map and sture the result in a list
    # convert the list to a tf tensor and send to predict method
    # result is a BATCH_SIZE length numpy array containing predictions
    
    # return model.predict(tf.convert_to_tensor(list(map(convert_fen_to_bitboard, boards))))
    return model.predict(tf.convert_to_tensor(input_data))