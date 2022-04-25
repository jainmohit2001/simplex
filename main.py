import os
import sys
import time

import cplex
import numpy as np


class Simplex:
    def __init__(self, filename: str, verbose: bool = False):
        """
        Simplex wrapper class.
        Algorithm used - 2-Phase Simplex method.

        Usage:
            simplex = Simplex(filename='input.lp', verbose=False)
            simplex.solve()
        """
        self.verbose = verbose
        self.start_time = time.time()
        self.end_time = time.time()
        self.cpx = self.get_equational_form(filename)
        self.iter = 0
        self.A = self.b = self.c = np.zeros(0)
        self.__initialize_matrices()
        self.feasible = True
        self.bounded = False
        self.optimal = False
        self.optimal_value = None
        self.optimal_sol = np.zeros(0)
        self.MAX_ITERATIONS = 10 ** 5
        self.INT_MAX = float('inf')

    def get_equational_form(self, filename) -> cplex.Cplex:
        if self.verbose:
            print("Convert problem to equational form with b>=0")
        cpx = cplex.Cplex(os.path.join(os.getcwd(), filename))  # Read input file using cplex library
        num_constraints = cpx.linear_constraints.get_num()  # Total number of constraints
        num_slack = 0  # Total number of slack/surplus variables

        # Converting problem to maximization problem if required
        if cpx.objective.sense[cpx.objective.get_sense()] == 'minimize':
            cpx.objective.set_sense(cpx.objective.sense.maximize)
            c = cpx.objective.get_linear()

            # Negating coefficients of the objective function
            cpx.objective.set_linear([(i, -1 * c[i]) for i in range(len(c))])

        for i in range(num_constraints):
            lc = cpx.linear_constraints.get_rows(i)  # example: SparsePair(ind=[0,2,5], val=[4,5,1])
            lc_name = cpx.linear_constraints.get_names(i)  # Name of the constraint
            rhs = cpx.linear_constraints.get_rhs(i)  # RHS value of the constraint

            # 'G' => Greater than or equal to
            # 'L' => Less than or equal to
            # 'E' => Equal to
            sense = cpx.linear_constraints.get_senses(i)

            # If rhs < 0, multiply the constraint with -1.
            # Updating coefficients, rhs value and changing the sense of the constraint.
            if rhs < 0:
                new_sp = cplex.SparsePair(ind=lc.unpack()[0], val=[-1 * i for i in lc.unpack()[1]])
                cpx.linear_constraints.set_linear_components(lc_name, new_sp)
                cpx.linear_constraints.set_rhs(lc_name, -1 * rhs)
                if sense == 'L':
                    cpx.linear_constraints.set_senses(lc_name, 'G')
                elif sense == 'G':
                    cpx.linear_constraints.set_senses(lc_name, 'L')

            sense = cpx.linear_constraints.get_senses(i)  # Updated sense
            if sense == 'L':
                # Adding slack with +1 coefficient
                num_slack += 1
                slack_name = f's{num_slack}'
                cpx.variables.add(obj=[0], lb=[0], ub=[cplex.infinity], names=[slack_name])
                cpx.linear_constraints.set_coefficients(lc_name, slack_name, 1)
            elif sense == 'G':
                # Adding surplus with -1 coefficient
                num_slack += 1
                slack_name = f's{num_slack}'
                cpx.variables.add(obj=[0], lb=[0], ub=[cplex.infinity], names=[slack_name])
                cpx.linear_constraints.set_coefficients(lc_name, slack_name, -1)

        if self.verbose:
            print("Total Slack/Surplus variable added : %s" % num_slack)
        return cpx

    def cplex_solver(self):
        """
        The LP solver provided by CPLEX API. Used as a reference and running time comparison.
        """
        start_time = time.time()
        self.cpx.set_problem_type(self.cpx.problem_type.LP)
        if not self.verbose:
            self.cpx.set_log_stream(None)
            self.cpx.set_results_stream(None)
            self.cpx.set_error_stream(None)
            self.cpx.set_warning_stream(None)
        self.cpx.parameters.lpmethod.set(self.cpx.parameters.lpmethod.values.primal)
        self.cpx.solve()
        print("cplex solver status :  %s" % self.cpx.solution.status[self.cpx.solution.get_status()])
        try:
            print("Objective value: %s" % self.cpx.solution.get_objective_value())
        except Exception as e:
            print("No solution provided by cplex solver")
        end_time = time.time()
        print("Total time taken by cplex solver in seconds: %s" % (end_time - start_time))

    def __initialize_matrices(self):
        """
        This function computes the initial matrices A, b, c from the auxiliary problem.
        """
        linear_constraints = self.cpx.linear_constraints.get_rows()
        num_constraints = self.cpx.linear_constraints.get_num()
        num_variables = self.cpx.variables.get_num()
        A = np.zeros((num_constraints, num_variables), dtype='float')

        # Using cplex library to get b and c
        b = np.array(self.cpx.linear_constraints.get_rhs())
        c = np.array(self.cpx.objective.get_linear())

        i = 0
        for constraint in linear_constraints:
            # The constraints are in SparsePair format specified by cplex
            indices = constraint.unpack()[0]
            values = constraint.unpack()[1]
            k = 0
            for j in indices:
                A[i][j] = values[k]
                k += 1
            i += 1
        self.A = A
        self.b = b
        self.c = c
        if self.verbose:
            print("Initialized Matrices: ")
            print("---------- A ----------")
            print(self.A)
            print("---------- b ----------")
            print(self.b)
            print("---------- c ----------")
            print(self.c)

    def get_tableau_phase_1(self) -> np.array:
        """
        This function uses the matrices A, b, c and computes the initial simplex tableau required for phase 1.
        """
        num_variables = len(self.c)
        num_artificial = len(self.A)  # Adding artificial variables to all the constraints for simplification

        # Top row t1, initialized to [0].
        # The first element is of type NaN and is used for book keeping reasons only.
        # The second element if the value of the auxiliary ost function, initialized to 0 (will be computed later)
        # The rest of the elements are the coefficients of the variables in the auxiliary cost function.
        t1 = np.hstack(([None], [0], [0] * num_variables, [0] * num_artificial))

        # Second row t2
        # The first element is of type NaN and is used for book keeping reasons only.
        # The second element if the value of the original cost function, initialized to 0
        # The rest of the elements are the coefficients of the variables in the auxiliary cost function.
        # This row is used to compute the row zero for phase 2.
        t2 = np.hstack(([None], [0], -1 * self.c, [0] * num_artificial))

        # Initial BFS for auxiliary LP, all the artificial variables  (zero based indexing)
        basis = np.array([0] * num_artificial)
        for i in range(0, len(basis)):
            basis[i] = num_variables + i
        A = self.A
        if not ((num_artificial + num_variables) == len(self.A[0])):
            # Adding identity matrix corresponding to the artificial variables
            B = np.identity(num_artificial)
            A = np.hstack((self.A, B))

        # The reset of the tableau consist of t3 with the following format
        # [basis] | [b] | [A]
        t3 = np.hstack((np.transpose([basis]), np.transpose([self.b]), A))
        tableau = np.vstack((t1, t2, t3))

        # Computing the top row of the tableau
        # Since the auxiliary objective is now W = Maximize -1 * (Sum of artificial variables)
        # After adding all the equational constraints and rearranging to get the above expression
        # We can see that the coefficient of variable xi is the sum of entries in its corresponding column in tableau
        # We add a minus sign because int the final objective auxiliary function
        for i in range(1, len(tableau[0]) - num_artificial):
            for j in range(2, len(tableau)):
                tableau[0, i] -= tableau[j, i]
        tableau = np.array(tableau, dtype='float')
        return tableau

    def get_tableau_phase2(self, tableau):
        """
        This function removes the row corresponding to the auxiliary function and returns the new tableau for phase 2.
        """
        tableau = np.delete(tableau, obj=[0], axis=0)
        return tableau

    def remove_av(self, tableau):
        """
        This function removes all the artificial variables from the auxiliary tableau.
        """
        av_ind = []
        tableau = np.delete(tableau, obj=np.s_[2 + len(self.c):], axis=1)
        for i in range(2, len(tableau)):
            # If the variable is an artificial variable
            if tableau[i, 0] > len(self.c) - 1:
                redundant = True  # This redundant variables signifies whether the row has all zeros or not
                for j in range(2, len(self.c) + 2):
                    if tableau[i, j] != 0:
                        redundant = False
                        break
                av_ind.append([i, redundant])

        if len(av_ind) == 0:
            if self.verbose:
                print("No artificial variables to remove")
            return tableau

        for key in av_ind:
            index, redundant = key
            if redundant:
                # If the variable is redundant then we can simply delete the corresponding row
                tableau = np.delete(tableau, obj=index, axis=0)
            else:
                # Otherwise we perform a pivot operation and remove this variable from the basis
                pivot_col = -1
                for i in range(2, len(tableau[0]) - 1):
                    if tableau[index, i] != 0:
                        pivot_col = i
                        break
                tableau = self.pivot_on(tableau, index, pivot_col)

                # pivot_col signifies the new entering variable
                tableau[index, 0] = pivot_col - 2
        return tableau

    def solve(self):
        """
        The main solve function that performs the simplex in 2 phases
        """
        tableau = self.get_tableau_phase_1()

        if self.verbose:
            print("----- Starting Phase 1 -----")

        tableau = self.simplex(tableau, starting_row=2)

        if not self.bounded:
            self.end_time = time.time()
            return

        if round(tableau[0, 1], 8) != 0:
            # If the auxiliary LP doesn't have an optimal solution with objective value = 0
            # then the original problem is infeasible
            self.feasible = False
            self.end_time = time.time()
            return

        if self.verbose:
            print("----- Starting Phase 2 -----")

        tableau = self.remove_av(tableau)
        tableau = self.get_tableau_phase2(tableau)

        # Rounding variables upto 8 decimals. Python has some issues with floating point operations.
        for i in range(len(tableau)):
            for j in range(len(tableau[0])):
                tableau[i][j] = round(tableau[i][j], 8)
        tableau = self.simplex(tableau, starting_row=1)

        if not self.bounded:
            self.end_time = time.time()
            return

        self.optimal_sol = np.array([0] * len(self.c), dtype=float)

        # Saving values of the variables that are in the final basis of the optimal solution
        for key in range(1, (len(tableau))):
            if tableau[key, 0] < len(self.c):
                self.optimal_sol[int(tableau[key, 0])] = tableau[key, 1]
        self.optimal_value = tableau[0, 1]
        self.end_time = time.time()

    def print_solution(self):
        print('Total time taken in seconds taken by 2 phase method : %s' % (self.end_time - self.start_time))
        if self.feasible:
            if self.bounded:
                print("Optimal value: %s" % self.optimal_value)
            else:
                print("Problem is unbounded!")
        else:
            print("Problem is infeasible!")

    def simplex(self, tableau, starting_row=2) -> np.array:
        if self.verbose:
            print("Starting Tableau")
            print(tableau)

        self.iter = 1

        while 1:
            if self.verbose:
                print(f"---- Iteration : {self.iter}")
                print(tableau)

            self.optimal = True
            for i in tableau[0, 2:]:
                if round(i, 8) < 0:
                    self.optimal = False
                    break

            if self.optimal:
                break
            if self.iter > self.MAX_ITERATIONS:
                break

            pivot_col = tableau[0, 2:].tolist().index(np.amin(tableau[0, 2:])) + 2

            self.bounded = False
            for element in tableau[starting_row:, pivot_col]:
                if round(element, 8) > 0:
                    self.bounded = True
                    break

            if not self.bounded:
                print("Unbounded; No solution.")
                return

            minimum = float('inf')
            pivot_row = -1
            for i in range(starting_row, len(tableau)):
                if round(tableau[i, pivot_col],8) > 0:
                    val = tableau[i, 1] / tableau[i, pivot_col]
                    if round(val, 8) < round(minimum, 8):
                        minimum = val
                        pivot_row = i

            pivot_element = round(tableau[pivot_row, pivot_col], 8)

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
        """
        This function performs the pivot on a given tableau with the given pivot_row and pivot_column
        """
        pivot = round(tableau[pivot_row, pivot_col], 8)
        if self.verbose:
            print("pivoting with (row,col) = (%s,%s) and pivot_value= %s" % (pivot_row, pivot_col, pivot))
        tableau[pivot_row, 1:] = tableau[pivot_row, 1:] / pivot

        for i in range(0, len(tableau)):
            if i != pivot_row:
                tableau[i, 1:] = np.round(
                    tableau[i, 1:] - (tableau[i, pivot_col] / tableau[pivot_row, pivot_col]) * tableau[pivot_row, 1:],
                    14)
        return tableau


def main():
    """
    Usage:
        python main.py test1.lp
        python main.py test2.lp -v
    """
    filename = sys.argv[1]
    verbose = '-v' in sys.argv
    try:
        simplex = Simplex(filename=filename, verbose=verbose)
        simplex.solve()
        simplex.print_solution()
        simplex.cplex_solver()
    except Exception as e:
        print(e)
        sys.exit("Some Error occurred")


if __name__ == '__main__':
    main()
