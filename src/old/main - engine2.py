# System Packages

# Third-Party Packages
import numpy as np

# My Packages
# import Util

# Project Packages
import engine2 as engine


if __name__ == '__main__':
    np.set_printoptions(
        linewidth = 9999,
        precision = 6,
        edgeitems = 10,
        threshold = 4000,
        suppress = True
    )
    
    # engine.set_tps(60)
    # engine.set_fps(60)
    
    # engine.set_game(game)
    # engine.set_input(engine.InputPygame())
    
    engine.run()
