from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

from xgbx import model
from sulfur import model_s


class HUAWEI_CUP17_ProblemB(FloatProblem):
    """ Problem ZDT1.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
    .. note:: Continuous problem having a convex Pareto front
    """

    def __init__(self, number_of_variables: int=31):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super().__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['x', 'y']

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        tmp_list = []
        tmp_list.append(solution.variables)
        data_n = np.array(tmp_list)
        solution.objectives[0] = model.function(data_n) #solution.variables[0]
        solution.objectives[1]=model_s.function(data_n)

        print(solution.variables)
        print("octane:%f" % solution.objectives[0])
        ss = model.function(data_n)
        print('sulfur:%f' % solution.objectives[1])


        return solution

    # def eval_h(self,solution):


    def get_name(self):
        return 'HUAWEI_CUP17_ProblemB'