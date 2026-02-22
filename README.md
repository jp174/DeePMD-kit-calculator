# DeePMD-kit-calculator
If you are using DeePMD-kit to train your MLIP's you might want to know how close is your trained model with the referenced data, this script will help you to calculate the diference per atom in meV betwen the predicted energy and the reference energy

first we import the next modules:    
from deepmd.infer import DeepPot
import numpy as np
