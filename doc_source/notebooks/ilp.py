from cereeberus import ReebGraph, MapperGraph, Interleave
import cereeberus.data.ex_mappergraphs as ex_mg

import matplotlib.pyplot as plt
import numpy as np

import pulp #for ILP optimization

# retrive all the function values of the mappers
def set_function_values(myInt):
    """
    Set the function values for the ILP optimization problem.
    
    Parameters
    
        myInt: Interleave object
        The assignment that we want to optimize.
        
        
    Returns
        list
            The function values of the two mappers.
    """
    
    func_vals = [5]
    return func_vals

def create_map_variables(myInt, prob):
    """
    Create the decision variables for the ILP optimization problem. These are Phi, Phi^n, Psi, and Psi^n. Also set the initial values of the decision variables.
    
    Parameters
    
        myInt: Interleave object
        The assignment that we want to optimize.
        
        prob: pulp.LpProblem
        The ILP optimization problem.
        
    Returns
        Dict
            The decision variables (Phi, Phi^n, Psi, Psi^n)."""
    
    # get the function values
    func_vals = set_function_values(myInt)

    # Initial maps
    # phi
    Phi = {
    'Phi_0V': myInt.phi('0','V'),
    'Phi_nV': myInt.phi('n','V'),
    'Phi_0E': myInt.phi('0','E'),
    'Phi_nE': myInt.phi('n','E')
    }
    # psi
    Psi = {
    'Psi_0V': myInt.psi('0','V'),
    'Psi_nV': myInt.psi('n','V'),
    'Psi_0E': myInt.psi('0','E'),
    'Psi_nE': myInt.psi('n','E')
    }

    # create the decision variables at the block level
    Phi_vars = {'V':{'0':{},'n':{}},'E':{'0':{},'n':{}}} # to store all the lp variables for phi
    Psi_vars = {'V':{'0':{},'n':{}},'E':{'0':{},'n':{}}} # to store all the lp variables for psi

    for i in func_vals:
       for j in ['0', 'n']:
              for k in ['V', 'E']:
                    # phi

                    # create variables for each block
                    n_rows = Phi['Phi_'+j+k][i].get_array().shape[0]
                    n_cols = Phi['Phi_'+j+k][i].get_array().shape[1]
                    Phi_vars[k][j][i] = pulp.LpVariable.dicts('Phi_'+j+k+'_'+str(i), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

                    # set the initial values
                    for a in range(n_rows):
                        for b in range(n_cols):
                             prob += Phi_vars[k][j][i][(a,b)] == Phi['Phi_'+j+k][i].get_array()[a][b]

                    # psi
                    n_rows = Psi['Psi_'+j+k][i].get_array().shape[0]
                    n_cols = Psi['Psi_'+j+k][i].get_array().shape[1]
                    Psi_vars[k][j][i] = pulp.LpVariable.dicts('Psi_'+j+k+'_'+str(i), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

                    # set the initial values
                    for a in range(n_rows):
                        for b in range(n_cols):
                             prob += Psi_vars[k][j][i][(a,b)] == Psi['Psi_'+j+k][i].get_array()[a][b]

    return {'Phi_vars':Phi_vars, 'Psi_vars':Psi_vars}

def create_other_decision_variables(myInt):
    """
    Create the other decision variables for the ILP optimization problem. These are the reparametrization variables.

    Parameters

        myInt: Interleave object
        The assignment that we want to optimize.
        
        prob: pulp.LpProblem
        The ILP optimization problem.
        
    Returns
        Dict
            The decision variables z, map_product."""

    # get the function values
    func_vals = set_function_values(myInt)

    # the map matrices
    Phi = {
    'Phi_0V': myInt.phi('0','V'),
    'Phi_nV': myInt.phi('n','V'),
    'Phi_0E': myInt.phi('0','E'),
    'Phi_nE': myInt.phi('n','E')
    }
    Psi = {
    'Psi_0V': myInt.psi('0','V'),
    'Psi_nV': myInt.psi('n','V'),
    'Psi_0E': myInt.psi('0','E'),
    'Psi_nE': myInt.psi('n','E')
    }

    # create the decision variables at the block level
    z_vars = {'V':{'FGF':{}, 'GFG':{}},'E':{'FGF':{}, 'GFG':{}}}
    map_product_vars = {'V':{'FGF':{}, 'GFG':{}},'E':{'FGF':{}, 'GFG':{}}}

    for i in func_vals:
        for k in ['V', 'E']:
            n_rows_1 = Psi['Psi_n'+k][i].get_array().shape[0]
            n_cols_1 = Phi['Phi_0'+k][i].get_array().shape[1]

            map_product_vars[k]['FGF'][i] = pulp.LpVariable.dicts('FGF_'+str(i), ((a, b) for a in range(n_rows_1) for b in range(n_cols_1)), cat='Integer')

            n_rows_2 = Phi['Phi_n'+k][i].get_array().shape[0]
            n_cols_2 = Psi['Psi_0'+k][i].get_array().shape[1]

            map_product_vars[k]['GFG'][i] = pulp.LpVariable.dicts('GFG_'+str(i), ((a, b) for a in range(n_rows_2) for b in range(n_cols_2)), cat='Integer')

            n_rowcol_1 = Psi['Psi_n'+k][i].get_array().shape[1]
            n_rowcol_2 = Phi['Phi_n'+k][i].get_array().shape[1]

            z_vars[k]['FGF'][i] = pulp.LpVariable.dicts('z_FGF_'+str(i), ((a, b, c) for a in range(n_rows_1) for b in range(n_rowcol_1) for c in range(n_cols_1)), cat='Binary')

            z_vars[k]['GFG'][i] = pulp.LpVariable.dicts('z_GFG_'+str(i), ((a, b, c) for a in range(n_rows_2) for b in range(n_rowcol_2) for c in range(n_cols_2)), cat='Binary')

    return {'z':z_vars, 'map_product':map_product_vars}


def set_objective_function(prob):
    """
    Set the objective function for the ILP optimization problem. This is created across all the blocks.
    
    Parameters
    
        prob: pulp.LpProblem
        The ILP optimization problem.
        
    Returns
        pulp.LpVariable
            The minmax variable."""
    
    # create the minmax variable
    minmax_var = pulp.LpVariable('minmax_var', lowBound=0, cat='Continuous')

    
    # define the objective function
    prob += minmax_var

    return minmax_var  


def set_triangle_constraints(myInt, prob):
    """
    Set the triangle constraints for the ILP optimization problem. These constraints are created block by block.
    
    Parameters
    
        myInt: Interleave object
        The assignment that we want to optimize.
        
        prob: pulp.LpProblem
        The ILP optimization problem.
        
    Returns
        None"""
    
    # get the function values
    func_vals = set_function_values(myInt)

    Phi = {
    'Phi_0V': myInt.phi('0','V'),
    'Phi_nV': myInt.phi('n','V'),
    'Phi_0E': myInt.phi('0','E'),
    'Phi_nE': myInt.phi('n','E')
    }
    Psi = {
    'Psi_0V': myInt.psi('0','V'),
    'Psi_nV': myInt.psi('n','V'),
    'Psi_0E': myInt.psi('0','E'),
    'Psi_nE': myInt.psi('n','E')
    }

    I = {
    # vert
    'I_F0V': myInt.I('F','0','V'),
    'I_FnV': myInt.I('F','n','V'),
    'I_G0V': myInt.I('G','0','V'),
    'I_GnV': myInt.I('G','n','V'),

    # edge
    'I_F0E': myInt.I('F','0','E'),
    'I_FnE': myInt.I('F','n','E'),
    'I_G0E': myInt.I('G','0','E'),
    'I_GnE': myInt.I('G','n','E')
    }
    

    D = {
    # vert
    'D_F2nV': myInt.D('F', '2n', 'V'),
    'D_G2nV': myInt.D('G', '2n', 'V'),

    # edge
    'D_F2nE': myInt.D('F', '2n', 'E'),
    'D_G2nE': myInt.D('G', '2n', 'E')
    }

    # phi
    Phi = {
    'Phi_0V': myInt.phi('0','V'),
    'Phi_nV': myInt.phi('n','V'),
    'Phi_0E': myInt.phi('0','E'),
    'Phi_nE': myInt.phi('n','E')
    }
    # psi
    Psi = {
    'Psi_0V': myInt.psi('0','V'),
    'Psi_nV': myInt.psi('n','V'),
    'Psi_0E': myInt.psi('0','E'),
    'Psi_nE': myInt.psi('n','E')
    }

    # map variables
    Phi_vars = create_map_variables(myInt, prob)['Phi_vars']
    Psi_vars = create_map_variables(myInt, prob)['Psi_vars']

    # other variables
    z_vars = create_other_decision_variables(myInt)['z']
    map_product_vars = create_other_decision_variables(myInt)['map_product']

    # minmax variable
    minmax_var = set_objective_function(prob)

    for i in func_vals:
        print(f"iteration {i}")
        for j in ['V', 'E']:
            for k in ['F', 'G']:
                # multiply inclusion matrices
                i_n_i_0 = I['I_'+k+'n'+j][i].get_array() @ I['I_'+k+'0'+j][i].get_array()
                

                # set maps
                if k == 'F':
                    map_1 = Psi['Psi_n'+j][i]
                    map_2 = Phi['Phi_0'+j][i]
                    map_1_vars = Psi_vars[j]['n'][i]
                    map_2_vars = Phi_vars[j]['0'][i]
                    map_product_var = map_product_vars[j]['FGF'][i]
                    z_var = z_vars[j]['FGF'][i]
                else:  
                    map_1 = Phi['Phi_n'+j][i]
                    map_2 = Psi['Psi_0'+j][i]
                    map_1_vars = Phi_vars[j]['n'][i]
                    map_2_vars = Psi_vars[j]['0'][i]
                    map_product_var = map_product_vars[j]['GFG'][i]
                    z_var = z_vars[j]['GFG'][i]
             
                # write d
                dist = D['D_'+k+'2n'+j][i].get_array()
                # set the dimensions 
                shape_m = dist.shape[0]
                shape_n = map_1.get_array().shape[1]
                shape_o = map_2.get_array().shape[1]

                
                #  constraint 1: loss is bigger than the absolute value of each matrix elements
                for a in range(shape_m):
                    prob += minmax_var >= pulp.lpSum(dist[b, a] * (i_n_i_0[a, d] - map_product_var[a, d]) for b in range(shape_m) for d in range(shape_o))
                    prob += -minmax_var <= pulp.lpSum(dist[b, a] * (i_n_i_0[a, d] - map_product_var[a, d]) for b in range(shape_m) for d in range(shape_o))

                

                # constraints 2: map_multiplication and z relation
                for a in range(shape_m):
                    for d in range(shape_o):
                        prob += map_product_var[a, d] == pulp.lpSum(z_var[a, c, d] for c in range(shape_n))


                # constraint 3: z is less than either of the map values and greater than sum of the map values minus 1
                for a in range(shape_m):
                    for c in range(shape_n):
                        for d in range(shape_o):
                            prob += z_var[a, c, d] <= map_1_vars[a, c]
                            prob += z_var[a, c, d] <= map_2_vars[c, d]
                            prob += z_var[a, c, d] >= map_1_vars[a, c] + map_2_vars[c, d] - 1

                # constraint 4: each column sums to 1
                for c in range(shape_n):
                    prob += pulp.lpSum(map_1_vars[a, c] for a in range(shape_m)) == 1

                for d in range(shape_o):
                    prob += pulp.lpSum(map_2_vars[c, d] for c in range(shape_n)) == 1



def create_ilp_problem(myInt):
    """
    Create the ILP optimization problem.
    
    Parameters
    
        myInt: Interleave object
        The assignment that we want to optimize.
        
    Returns
    
        prob: pulp.LpProblem
        The ILP optimization problem.
    """
    
    prob = pulp.LpProblem("Interleave Optimization Problem", pulp.LpMinimize)
    
    # Create and initialize decision matrices
    create_map_variables(myInt, prob)

    # create other decision variables
    create_other_decision_variables(myInt)
    
    # Set the objective function
    set_objective_function(prob)
    
    # Set the constraints
    set_triangle_constraints(myInt, prob)

    from cereeberus import ReebGraph, MapperGraph, Interleave
import cereeberus.data.ex_mappergraphs as ex_mg

import matplotlib.pyplot as plt
import numpy as np

import pulp #for ILP optimization

# retrive all the function values of the mappers
def set_function_values(myInt):
    """
    Set the function values for the ILP optimization problem.
    
    Parameters
    
        myInt: Interleave object
        The assignment that we want to optimize.
        
        
    Returns
        list
            The function values of the two mappers.
    """
    
    func_vals = myInt.all_func_vals()

    return func_vals

def create_map_variables(myInt, prob):
    """
    Create the decision variables for the ILP optimization problem. These are Phi, Phi^n, Psi, and Psi^n. Also set the initial values of the decision variables.
    
    Parameters
    
        myInt: Interleave object
        The assignment that we want to optimize.
        
        prob: pulp.LpProblem
        The ILP optimization problem.
        
    Returns
        Dict
            The decision variables (Phi, Phi^n, Psi, Psi^n)."""
    
    # get the function values
    func_vals = set_function_values(myInt)

    # Initial maps
    # phi
    Phi = {
    'Phi_0V': myInt.phi('0','V'),
    'Phi_nV': myInt.phi('n','V'),
    'Phi_0E': myInt.phi('0','E'),
    'Phi_nE': myInt.phi('n','E')
    }
    # psi
    Psi = {
    'Psi_0V': myInt.psi('0','V'),
    'Psi_nV': myInt.psi('n','V'),
    'Psi_0E': myInt.psi('0','E'),
    'Psi_nE': myInt.psi('n','E')
    }

    # create the decision variables at the block level
    Phi_vars = {'V':{'0':{},'n':{}},'E':{'0':{},'n':{}}} # to store all the lp variables for phi
    Psi_vars = {'V':{'0':{},'n':{}},'E':{'0':{},'n':{}}} # to store all the lp variables for psi

    for i in func_vals:
       for j in ['0', 'n']:
              for k in ['V', 'E']:
                    # phi

                    # create variables for each block
                    n_rows = Phi['Phi_'+j+k][i].get_array().shape[0]
                    n_cols = Phi['Phi_'+j+k][i].get_array().shape[1]
                    Phi_vars[k][j][i] = pulp.LpVariable.dicts('Phi_'+j+k+'_'+str(i), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

                    # set the initial values
                    for a in range(n_rows):
                        for b in range(n_cols):
                             prob += Phi_vars[k][j][i][(a,b)] == Phi['Phi_'+j+k][i].get_array()[a][b]

                    # psi
                    n_rows = Psi['Psi_'+j+k][i].get_array().shape[0]
                    n_cols = Psi['Psi_'+j+k][i].get_array().shape[1]
                    Psi_vars[k][j][i] = pulp.LpVariable.dicts('Psi_'+j+k+'_'+str(i), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

                    # set the initial values
                    for a in range(n_rows):
                        for b in range(n_cols):
                             prob += Psi_vars[k][j][i][(a,b)] == Psi['Psi_'+j+k][i].get_array()[a][b]

    return {'Phi_vars':Phi_vars, 'Psi_vars':Psi_vars}

def create_other_decision_variables(myInt):
    """
    Create the other decision variables for the ILP optimization problem. These are the reparametrization variables.

    Parameters

        myInt: Interleave object
        The assignment that we want to optimize.
        
        prob: pulp.LpProblem
        The ILP optimization problem.
        
    Returns
        Dict
            The decision variables z, map_product."""

    # get the function values
    func_vals = set_function_values(myInt)

    # the initial map matrices
    Phi = {
    'Phi_0V': myInt.phi('0','V'),
    'Phi_nV': myInt.phi('n','V'),
    'Phi_0E': myInt.phi('0','E'),
    'Phi_nE': myInt.phi('n','E')
    }
    Psi = {
    'Psi_0V': myInt.psi('0','V'),
    'Psi_nV': myInt.psi('n','V'),
    'Psi_0E': myInt.psi('0','E'),
    'Psi_nE': myInt.psi('n','E')
    }

    # create the decision variables at the block level
    z_vars = {'V':{'FGF':{}, 'GFG':{}},'E':{'FGF':{}, 'GFG':{}}}
    map_product_vars = {'V':{'FGF':{}, 'GFG':{}},'E':{'FGF':{}, 'GFG':{}}}

    for i in func_vals:
        for k in ['V', 'E']:
            n_rows_1 = Psi['Psi_n'+k][i].get_array().shape[0]
            n_cols_1 = Phi['Phi_0'+k][i].get_array().shape[1]

            map_product_vars[k]['FGF'][i] = pulp.LpVariable.dicts('FGF_'+str(i), ((a, b) for a in range(n_rows_1) for b in range(n_cols_1)), cat='Integer')

            n_rows_2 = Phi['Phi_n'+k][i].get_array().shape[0]
            n_cols_2 = Psi['Psi_0'+k][i].get_array().shape[1]

            map_product_vars[k]['GFG'][i] = pulp.LpVariable.dicts('GFG_'+str(i), ((a, b) for a in range(n_rows_2) for b in range(n_cols_2)), cat='Integer')

            n_rowcol_1 = Psi['Psi_n'+k][i].get_array().shape[1]
            n_rowcol_2 = Phi['Phi_n'+k][i].get_array().shape[1]

            z_vars[k]['FGF'][i] = pulp.LpVariable.dicts('z_FGF_'+str(i), ((a, b, c) for a in range(n_rows_1) for b in range(n_rowcol_1) for c in range(n_cols_1)), cat='Binary')

            z_vars[k]['GFG'][i] = pulp.LpVariable.dicts('z_GFG_'+str(i), ((a, b, c) for a in range(n_rows_2) for b in range(n_rowcol_2) for c in range(n_cols_2)), cat='Binary')

    return {'z':z_vars, 'map_product':map_product_vars}


def set_objective_function(prob):
    """
    Set the objective function for the ILP optimization problem. This is created across all the blocks.
    
    Parameters
    
        prob: pulp.LpProblem
        The ILP optimization problem.
        
    Returns
        pulp.LpVariable
            The minmax variable."""
    
    # create the minmax variable
    minmax_var = pulp.LpVariable('minmax_var', lowBound=0, cat='Continuous')

    
    # define the objective function
    prob += minmax_var

    return minmax_var  


def set_triangle_constraints(myInt, prob):
    """
    Set the triangle constraints for the ILP optimization problem. These constraints are created block by block.
    
    Parameters
    
        myInt: Interleave object
        The assignment that we want to optimize.
        
        prob: pulp.LpProblem
        The ILP optimization problem.
        
    Returns
        None"""
    
    # get the function values
    func_vals = set_function_values(myInt)

    Phi = {
    'Phi_0V': myInt.phi('0','V'),
    'Phi_nV': myInt.phi('n','V'),
    'Phi_0E': myInt.phi('0','E'),
    'Phi_nE': myInt.phi('n','E')
    }
    Psi = {
    'Psi_0V': myInt.psi('0','V'),
    'Psi_nV': myInt.psi('n','V'),
    'Psi_0E': myInt.psi('0','E'),
    'Psi_nE': myInt.psi('n','E')
    }

    I = {
    # vert
    'I_F0V': myInt.I('F','0','V'),
    'I_FnV': myInt.I('F','n','V'),
    'I_G0V': myInt.I('G','0','V'),
    'I_GnV': myInt.I('G','n','V'),

    # edge
    'I_F0E': myInt.I('F','0','E'),
    'I_FnE': myInt.I('F','n','E'),
    'I_G0E': myInt.I('G','0','E'),
    'I_GnE': myInt.I('G','n','E')
    }
    

    D = {
    # vert
    'D_F2nV': myInt.D('F', '2n', 'V'),
    'D_G2nV': myInt.D('G', '2n', 'V'),

    # edge
    'D_F2nE': myInt.D('F', '2n', 'E'),
    'D_G2nE': myInt.D('G', '2n', 'E')
    }

    # phi
    Phi = {
    'Phi_0V': myInt.phi('0','V'),
    'Phi_nV': myInt.phi('n','V'),
    'Phi_0E': myInt.phi('0','E'),
    'Phi_nE': myInt.phi('n','E')
    }
    # psi
    Psi = {
    'Psi_0V': myInt.psi('0','V'),
    'Psi_nV': myInt.psi('n','V'),
    'Psi_0E': myInt.psi('0','E'),
    'Psi_nE': myInt.psi('n','E')
    }

    # map variables
    Phi_vars = create_map_variables(myInt, prob)['Phi_vars']
    Psi_vars = create_map_variables(myInt, prob)['Psi_vars']

    # other variables
    z_vars = create_other_decision_variables(myInt)['z']
    map_product_vars = create_other_decision_variables(myInt)['map_product']

    # minmax variable
    minmax_var = set_objective_function(prob)

    for i in func_vals:
        for j in ['V', 'E']:
            for k in ['F', 'G']:
                # multiply inclusion matrices
                i_n_i_0 = I['I_'+k+'n'+j][i].get_array() @ I['I_'+k+'0'+j][i].get_array()
                

                # set maps
                if k == 'F':
                    map_1 = Psi['Psi_n'+j][i]
                    map_2 = Phi['Phi_0'+j][i]
                    map_1_vars = Psi_vars[j]['n'][i]
                    map_2_vars = Phi_vars[j]['0'][i]
                    map_product_var = map_product_vars[j]['FGF'][i]
                    z_var = z_vars[j]['FGF'][i]
                else:  
                    map_1 = Phi['Phi_n'+j][i]
                    map_2 = Psi['Psi_0'+j][i]
                    map_1_vars = Phi_vars[j]['n'][i]
                    map_2_vars = Psi_vars[j]['0'][i]
                    map_product_var = map_product_vars[j]['GFG'][i]
                    z_var = z_vars[j]['GFG'][i]
             
                # write d
                dist = D['D_'+k+'2n'+j][i].get_array()
                # set the dimensions 
                shape_m = dist.shape[0]
                shape_n = map_1.get_array().shape[1]
                shape_o = map_2.get_array().shape[1]

                
                #  constraint 1: loss is bigger than the absolute value of each matrix elements
                for a in range(shape_m):
                    prob += minmax_var >= pulp.lpSum(dist[b, a] * (i_n_i_0[a, d] - map_product_var[a, d]) for b in range(shape_m) for d in range(shape_o))
                    prob += -minmax_var <= pulp.lpSum(dist[b, a] * (i_n_i_0[a, d] - map_product_var[a, d]) for b in range(shape_m) for d in range(shape_o))

                

                # constraints 2: map_multiplication and z relation
                for a in range(shape_m):
                    for d in range(shape_o):
                        prob += map_product_var[a, d] == pulp.lpSum(z_var[a, c, d] for c in range(shape_n))


                # constraint 3: z is less than either of the map values and greater than sum of the map values minus 1
                for a in range(shape_m):
                    for c in range(shape_n):
                        for d in range(shape_o):
                            prob += z_var[a, c, d] <= map_1_vars[a, c]
                            prob += z_var[a, c, d] <= map_2_vars[c, d]
                            prob += z_var[a, c, d] >= map_1_vars[a, c] + map_2_vars[c, d] - 1

                # constraint 4: each column sums to 1
                for c in range(shape_n):
                    prob += pulp.lpSum(map_1_vars[a, c] for a in range(shape_m)) == 1

                for d in range(shape_o):
                    prob += pulp.lpSum(map_2_vars[c, d] for c in range(shape_n)) == 1



def create_ilp_problem(myInt):
    """
    Create the ILP optimization problem.
    
    Parameters
    
        myInt: Interleave object
        The assignment that we want to optimize.
        
    Returns
    
        prob: pulp.LpProblem
        The ILP optimization problem.
    """
    
    prob = pulp.LpProblem("Interleave Optimization Problem", pulp.LpMinimize)
    

    # Set the objective function
    set_objective_function(prob)
    
    # Set the constraints
    set_triangle_constraints(myInt, prob)

    pulp.LpStatus[prob.status]

    prob.writeLP("model.lp")  # Write the model in LP format


    # solve the problem
    prob.solve()

    print("status:", pulp.LpStatus[prob.status])
    
    

def solve_ilp(myInt):

    # function values
    func_vals = myInt.all_func_vals()

    # create the ILP problem
    prob = pulp.LpProblem("Interleave Optimization Problem", pulp.LpMinimize)

    # create empty dictionaries to store the decision variables
    Phi_vars = {block : {thickening:{obj_type:{} for obj_type in ['V', 'E']} for thickening in ['0', 'n']} for block in func_vals}
    Psi_vars = {block : {thickening:{obj_type:{} for obj_type in ['V', 'E']} for thickening in ['0', 'n']} for block in func_vals}


    # create the decision variables
    for block in func_vals:
        for thickening in ['0', 'n']:
            for obj_type in ['V', 'E']:
                
                # create lp variables for phi
                n_rows = myInt.phi(thickening, obj_type)[block].get_array().shape[0]
                n_cols = myInt.phi(thickening, obj_type)[block].get_array().shape[1]

                Phi_vars[block][thickening][obj_type] = pulp.LpVariable.dicts('Phi_'+thickening+obj_type+'_'+str(block), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

                # set the initial values
                for a in range(n_rows):
                    for b in range(n_cols):
                        prob += Phi_vars[block][thickening][obj_type][(a,b)] == myInt.phi(thickening, obj_type)[block].get_array()[a][b]


                # create lp variables for psi
                n_rows = myInt.psi(thickening, obj_type)[block].get_array().shape[0]
                n_cols = myInt.psi(thickening, obj_type)[block].get_array().shape[1]

                Psi_vars[block][thickening][obj_type] = pulp.LpVariable.dicts('Psi_'+thickening+obj_type+'_'+str(block), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

                # set the initial values
                for a in range(n_rows):
                    for b in range(n_cols):
                        prob += Psi_vars[block][thickening][obj_type][(a,b)] == myInt.psi(thickening, obj_type)[block].get_array()[a][b]

    # create the other decision variables
    z_vars = {block :  {starting_map: {obj_type: {} for obj_type in ['V', 'E']} for starting_map in ['F', 'G']} for block in func_vals}

    map_product_vars = {block : {starting_map: {obj_type: {} for obj_type in ['V', 'E']} for starting_map in ['F', 'G']} for block in func_vals}

    for block in func_vals:
        for obj_type in ['V', 'E']:
            for starting_map in ['F', 'G']:
                
                if starting_map == 'F':
                    n_rows_1 = myInt.psi('n', obj_type)[block].get_array().shape[0]
                    n_cols_1 = myInt.phi('0', obj_type)[block].get_array().shape[1]
                    
                    # set the map product variables
                    map_product_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts(starting_map+'_'+obj_type+'_'+str(block), ((a, b) for a in range(n_rows_1) for b in range(n_cols_1)), cat='Integer')

                    n_rowcol_1 = myInt.psi('n', obj_type)[block].get_array().shape[1]

                    z_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts('z_'+starting_map+'_'+obj_type+'_'+str(block), ((a, b, c) for a in range(n_rows_1) for b in range(n_rowcol_1) for c in range(n_cols_1)), cat='Binary')
                    
                else:
                    n_rows_1 = myInt.phi('n', obj_type)[block].get_array().shape[0]
                    n_cols_1 = myInt.psi('0', obj_type)[block].get_array().shape[1]

                    # set the map product variables
                    map_product_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts(starting_map+'_'+obj_type+'_'+str(block), ((a, b) for a in range(n_rows_1) for b in range(n_cols_1)), cat='Integer')

                    n_rowcol_2 = myInt.phi('n', obj_type)[block].get_array().shape[1]

                    z_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts('z_'+starting_map+'_'+obj_type+'_'+str(block), ((a, b, c) for a in range(n_rows_1) for b in range(n_rowcol_2) for c in range(n_cols_1)), cat='Binary')

    # create the minmax variable
    minmax_var = pulp.LpVariable('minmax_var', lowBound=0, cat='Continuous')

    # Set the objective function
    prob += minmax_var
    
    # create the constraints

    for block in func_vals:
        for obj_type in ['V', 'E']:
            for starting_map in ['F', 'G']:
                # multiply inclusion matrices
                i_n_i_0 = myInt.I(starting_map, 'n', obj_type)[block].get_array() @ myInt.I(starting_map, '0', obj_type)[block].get_array()

                # write dist matrix for easier reference
                dist = myInt.D(starting_map, '2n', obj_type)[block].get_array()

                # set the dimensions
                shape_m = dist.shape[0]
                if starting_map == 'F':
                    shape_n = myInt.psi('n', obj_type)[block].get_array().shape[1]
                    shape_o = myInt.phi('0', obj_type)[block].get_array().shape[1]
                else:
                    shape_n = myInt.phi('n', obj_type)[block].get_array().shape[1]
                    shape_o = myInt.psi('0', obj_type)[block].get_array().shape[1]

                #  constraint 1: loss is bigger than the absolute value of each matrix elements
            
                for  h in range(shape_m):
                    prob += minmax_var >= pulp.lpSum(dist[i,h] * (i_n_i_0[h,k] - map_product_vars[block][starting_map][obj_type][h,k]) for i in range(shape_m) for k in range(shape_o))
                    prob += -minmax_var <= pulp.lpSum(dist[i,h] * (i_n_i_0[h,k] - map_product_vars[block][starting_map][obj_type][h,k]) for i in range(shape_m) for k in range(shape_o))

                # constraint 2: map_multiplication and z relation
                for i in range(shape_m):
                    for k in range(shape_o):
                        prob += map_product_vars[block][starting_map][obj_type][i,k] == pulp.lpSum(z_vars[block][starting_map][obj_type][i,j,k] for j in range(shape_n))

                # constraint 3: z is less than either of the map values and greater than sum of the map values minus 1
                for i in range(shape_m):
                    for j in range(shape_n):
                        for k in range(shape_o):
                            if starting_map == 'F':
                                prob += z_vars[block][starting_map][obj_type][i,j,k] <= Psi_vars[block]['n'][obj_type][i,j]
                                prob += z_vars[block][starting_map][obj_type][i,j,k] <= Phi_vars[block]['0'][obj_type][j,k]
                                prob += z_vars[block][starting_map][obj_type][i,j,k] >= Psi_vars[block]['n'][obj_type][i,j] + Phi_vars[block]['0'][obj_type][j,k] - 1
                            else:
                                prob += z_vars[block][starting_map][obj_type][i,j,k] <= Phi_vars[block]['n'][obj_type][i,j]
                                prob += z_vars[block][starting_map][obj_type][i,j,k] <= Psi_vars[block]['0'][obj_type][j,k]
                                prob += z_vars[block][starting_map][obj_type][i,j,k] >= Phi_vars[block]['n'][obj_type][i,j] + Psi_vars[block]['0'][obj_type][j,k] - 1

                # constraint 4: each column sums to 1
                if starting_map == 'F':
                    for j in range(shape_n):
                        prob += pulp.lpSum(Psi_vars[block]['n'][obj_type][i,j] for i in range(shape_m)) == 1

                    for k in range(shape_o):
                        prob += pulp.lpSum(Phi_vars[block]['0'][obj_type][j,k] for j in range(shape_n)) == 1

                else:
                    for j in range(shape_n):
                        prob += pulp.lpSum(Phi_vars[block]['n'][obj_type][i,j] for i in range(shape_m)) == 1

                    for k in range(shape_o):
                        prob += pulp.lpSum(Psi_vars[block]['0'][obj_type][j,k] for j in range(shape_n)) == 1

                
                
    # solve the problem
    prob.solve()

    # print the loss
    print(f"Loss: {pulp.value(minmax_var)}")

    prob.writeLP("model.lp")  # Write the model in LP format