import read_info
import numpy as np
import os
import pandas as pd

if __name__ == "__main__":
    # 1) Read in all of the user-defined config parameters into a class, called Params.
    Params = read_info.ParamModel()