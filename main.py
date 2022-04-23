import os
from pprint import pprint

import cplex
import numpy as np


class Simplex:
    def __init__(self, filename: str, verbose=False):

        self.cpx = cplex.Cplex(os.path.join(os.getcwd(), filename))
        self.sense = self.cpx.objective.sense[self.cpx.objective.get_sense()]
        self.iter = 0
        self.A = self.b = self.c = np.empty(0)
        self.__initialize_matrices()
        self.verbose = verbose
        self.optimal = False
        self.optimal_value = None
        self.feasible = True
        self.bounded = False
        self.x = np.array(0)
        self.MAX_ITERATIONS = 10 ** 5
        self.int_max = float('inf')

    def cplex_solver(self):
        self.cpx.set_problem_type(self.cpx.problem_type.LP)
        self.cpx.solve()
        self.cpx.parameters.lpmethod.set(self.cpx.parameters.lpmethod.values.primal)
        print(self.cpx.solution.get_objective_value())
        print(self.cpx.solution.status[self.cpx.solution.get_status()])

    def print_tableau(self, tableau):
        pprint(tableau)

    def __initialize_matrices(self):
        # The Linear Constraints come in SparsePair format
        linear_constraints = self.cpx.linear_constraints.get_rows()

        num_constraints = self.cpx.linear_constraints.get_num()
        num_variables = self.cpx.variables.get_num()
        A = np.empty((num_constraints, num_variables))
        b = np.array(self.cpx.linear_constraints.get_rhs())
        c = np.array(self.cpx.objective.get_linear())
        i = 0
        for constraint in linear_constraints:
            # Using the indices and the corresponding values from SparsePair to populate a row
            indices = constraint.unpack()[0]
            values = constraint.unpack()[1]
            k = 0
            for j in indices:
                A[i][j] = values[k]
                k += 1
            i += 1

        if self.sense == 'minimize':
            c = -1 * c
        self.A = A
        self.b = b
        self.c = c

    def create_tableau_phase_1(self) -> np.array:
        num_variables = len(self.c)  # size of coefficient matrix c from objective function
        num_artificial = len(self.A)  # This is same as number of constraints

        # Top row
        # Left most element - NaN type and will not be used
        # The second element - Current cost (initialized = 0)
        # Followed by cost function coefficients including artificial variables.
        # t1 = np.hstack(([None], [0], [0] * num_variables, [0] * num_artificial))  # Top row
        t1 = np.hstack(([None], [0], -1 * self.c, [0] * num_artificial))  # Top row
        basis = np.array([0] * num_artificial)
        for i in range(0, len(basis)):
            basis[i] = num_variables + i
        A = self.A
        if not ((num_artificial + num_variables) == len(self.A[0])):
            B = np.identity(num_artificial)
            A = np.hstack((self.A, B))
        t2 = np.hstack((np.transpose([basis]), np.transpose([self.b]), A))
        tableau = np.vstack((t1, t2))
        # for i in range(1, len(tableau[0]) - num_artificial):
        #     for j in range(1, len(tableau)):
        #         if self.sense == "minimize":
        #             tableau[0, i] -= tableau[j, i]
        #         else:
        #             tableau[0, i] += tableau[j, i]
        tableau = np.array(tableau, dtype='float')
        return tableau

    def get_tableau_phase2(self, tableau):
        self.print_tableau(tableau)
        for i in range(0, len(self.c) - 1):
            tableau[0, i + 2] = self.c[i]

        tableau[0, 1] = 0

        a = np.empty(0)
        for value in tableau[0, 1:]:
            a = np.append(a, value)

        for j in range(1, len(tableau)):
            for i in range(1, len(tableau[0])):
                if self.sense == "minimum":
                    a[i - 1] -= tableau[0, int(tableau[j, 0] + 2)] * tableau[j, i]
                else:
                    a[i - 1] += tableau[0, int(tableau[j, 0] + 2)] * tableau[j, i]

        tableau[0, 1:] = a

        return tableau

    def drive_out_av(self, tableau):
        av_present = False
        av_ind = []

        # remove slack variables
        tableau = np.delete(tableau, obj=np.s_[2 + len(self.c):], axis=1)

        # redundant row removal
        for i in range(1, len(tableau)):
            if tableau[i, 0] > len(self.c) - 1:

                redundant = True
                for j in range(2, len(self.c) + 2):
                    if tableau[i, j] != 0:
                        redundant = False
                        av_ind.append([i, False])
                        break
                av_ind.append([i, redundant])

        if len(av_ind) == 0:
            print("\nNo artificial variables to drive out\n")
            return tableau

        for key in av_ind:
            index, redundant = key
            if redundant:
                tableau = np.delete(tableau, obj=index, axis=0)
                av_ind.remove(key)
            else:
                pivot_col = -1
                for i in range(2, len(tableau[0]) - 1):
                    if tableau[index, i] != 0:
                        pivot_col = i
                        break
                tableau = self.pivot_on(tableau, index, pivot_col)

        return tableau

    def solve_1_phase(self):
        tableau = self.create_tableau_phase_1()
        print("Phase 1:\n")
        tableau = self.simplex(tableau)

        if not self.bounded:
            return

        self.x = np.array([0] * len(self.c), dtype=float)

        # save coefficients
        for key in range(1, (len(tableau))):
            if tableau[key, 0] < len(self.c):
                self.x[int(tableau[key, 0])] = tableau[key, 1]

        self.optimal_value = tableau[0, 1]

    def solve_2_phase(self):
        tableau = self.create_tableau_phase_1()
        print("Phase 1:\n")
        tableau = self.simplex(tableau)

        if not self.bounded:
            return

        if tableau[0, 1] != 0:
            self.feasible = False
            print("Problem Infeasible; No Solution")
            return

        print("Phase 2:\n")

        tableau = self.drive_out_av(tableau)

        tableau = self.get_tableau_phase2(tableau)

        tableau = self.simplex(tableau)

        if not self.bounded:
            return

        self.x = np.array([0] * len(self.c), dtype=float)

        # save coefficients
        for key in range(1, (len(tableau))):
            if tableau[key, 0] < len(self.c):
                self.x[int(tableau[key, 0])] = tableau[key, 1]

        self.optimal_value = -1 * tableau[0, 1]

    def print_solution(self):
        if self.feasible:
            if self.bounded:
                print("Coefficients: ")
                print(self.x)
                print("Optimal value: ")
                pprint(self.optimal_value)
            else:
                print("Problem Unbounded; No Solution")
        else:
            print("Problem Infeasible; No Solution")

    def simplex(self, tableau) -> np.array:
        if self.verbose:
            print("Starting Tableau")
            self.print_tableau(tableau)

        self.iter = 1

        while 1:
            if self.verbose:
                print(f"---- Iteration : {self.iter}")
                self.print_tableau(tableau)

            if self.sense == 'maximum':
                for i in tableau[0, 2:]:
                    if i > 0:
                        self.optimal = False
                        break
                    self.optimal = True
            else:
                for i in tableau[0, 2:]:
                    if i < 0:
                        self.optimal = False
                        break
                    self.optimal = True

            if self.optimal:
                break
            if self.iter > self.MAX_ITERATIONS:
                break
            if self.sense == 'minimum':
                pivot_col = tableau[0, 2:].tolist().index(np.amax(tableau[0, 2:])) + 2
            else:
                pivot_col = tableau[0, 2:].tolist().index(np.amin(tableau[0, 2:])) + 2

            self.bounded = False
            for element in tableau[1:, pivot_col]:
                if element != 0:
                    self.bounded = True

            if not self.bounded:
                print("Unbounded; No solution.")
                return

            minimum = float('inf')
            pivot_row = -1
            for i in range(1, len(tableau)):
                if tableau[i, pivot_col] > 0:
                    val = tableau[i, 1] / tableau[i, pivot_col]
                    if val < minimum:
                        minimum = val
                        pivot_row = i

            pivot_element = tableau[pivot_row, pivot_col]

            if self.verbose:
                print("Pivot Column:", pivot_col)
                print("Pivot Row:", pivot_row)
                print("Pivot Element: ", pivot_element)
                print("Entering variable " + str(pivot_col - 1))
                print("Leaving variable " + str(int(tableau[pivot_row, 0] + 1)))

            tableau = self.pivot_on(tableau, pivot_col=pivot_col, pivot_row=pivot_row)
            tableau[pivot_row, 0] = pivot_col - 2
            self.iter += 1
        return tableau

    def pivot_on(self, tableau, pivot_row, pivot_col) -> np.array:
        pivot = tableau[pivot_row, pivot_col]
        tableau[pivot_row, 1:] = tableau[pivot_row, 1:] / pivot

        # pivot other rows
        for i in range(0, len(tableau)):
            if i != pivot_row:
                mult = tableau[i, pivot_col] / tableau[pivot_row, pivot_col]
                tableau[i, 1:] = tableau[i, 1:] - mult * tableau[pivot_row, 1:]

        return tableau


def main():
    simplex = Simplex(filename='test1.lp', verbose=True)
    simplex.solve_1_phase()
    simplex.print_solution()

    simplex.cplex_solver()


if __name__ == '__main__':
    main()
