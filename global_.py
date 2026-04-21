"""
some global variables
"""
task :int = None # current batch task id

TP_enable:bool = None # None means not set yet. should be set in imports.py
rank_:int = None
moduleName_2_adaRank:dict = {} # adaptive rank for each shared+LoRA module

