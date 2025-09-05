# from cereeberus import ReebGraph, MapperGraph, Interleave
from .labeled_blocks import LabeledBlockMatrix as LBM
from .labeled_blocks import LabeledMatrix as LM
# import cereeberus.data.ex_mappergraphs as ex_mg

import matplotlib.pyplot as plt
import numpy as np

import pulp #for ILP optimization



# function to build the phi and psi matrices after the ILP optimization
def build_map_matrices(myAssgn, map_results):
    """
    Function to build the map matrices after the ILP optimization. The end result is a dictionary with keys
    - 'phi_0_V', 'phi_0_E'
    - 'phi_n_V', 'phi_n_E' 
    - 'psi_0_V', 'psi_0_E'
    - 'psi_n_V', 'psi_n_E'.
    Each key corresponds to a LabeledBlockMatrix containing the respective map matrices for the given thickening. 
    
    Parameters:
        myAssgn (Assignment): the Assignment object
        map_results (dict): the dictionary containing the results of the ILP optimization from the solve_ilp function

    Returns:
        dict: the dictionary containing the final map matrices
    """

    # get the function values
    func_vals = myAssgn.all_func_vals()
    
    # create a dictionary to store the final matrices 
    final_LBMs = {}

    for thickening in ['0', 'n']:
        for obj_type in ['V', 'E']:
            for map_type in ['phi', 'psi']:
                
                # Gets the full LBM for the relevant interleaving map
                M = myAssgn.get_interleaving_map(maptype=map_type, key=thickening, obj_type=obj_type)
                
                for i in M.get_all_block_indices():
                    block_shape = M[i].array.shape 
                    
                    # Reset to zero 
                    M[i].array = np.zeros(block_shape)

                    # Set the values according to the ILP optimization results
                    for a in range(block_shape[0]):
                        for b in range(block_shape[1]):
                            # Get the relevant value from the ILP optimization results
                            if a<0 or b<0: 
                                print(f"Warning: a={a}, b={b} for block {i} in {map_type}_{thickening}_{obj_type}.")
                            try: 
                                M[i].array[a,b] = pulp.value(map_results[map_type+'_vars'][i][thickening][obj_type][(a,b)])
                            except KeyError:
                                # TODO: This makes me nervous, but I think the issue is just that there are empty rows/columns that we want to skip
                                # print(f"KeyError: {map_type+'_vars'} for block {i} in {map_type}_{thickening}_{obj_type}.")
                                continue
                            
                
                output_key = f"{map_type}_{thickening}_{obj_type}" 
                final_LBMs[output_key] = M

    return final_LBMs
    
##------------------- ILP Optimization -------------------##

def solve_ilp(myAssgn, pulp_solver = None, verbose=False):

    """
    Function to solve the ILP optimization problem for interleaving maps. The function creates a linear programming problem using the PuLP library and solves it to find the optimal interleaving maps.
    
    Parameters:
        myAssgn (Assignment): the Assignment object containing the interleaving maps and other relevant data
        pulp_solver (pulp.LpSolver): the solver to use for the ILP optimization. If None, the default solver is used.
        verbose (bool): whether to print the optimization status and results
        
    Returns:
        tuple: a tuple containing the final interleaving maps (as a dictionary of LabeledBlockMatrices) and the optimized loss value
    """
    # function values
    func_vals = myAssgn.all_func_vals()

    # create the ILP problem
    prob = pulp.LpProblem("Interleave_Optimization_Problem", pulp.LpMinimize)

    # create empty dictionaries to store the decision variables
    phi_vars = {block : {thickening:{obj_type:{} for obj_type in ['V', 'E']} for thickening in ['0', 'n']} for block in func_vals}
    psi_vars = {block : {thickening:{obj_type:{} for obj_type in ['V', 'E']} for thickening in ['0', 'n']} for block in func_vals}


    # create the decision variables (NOTE: these are all the decision variables for all the diagrams)
    for thickening in ['0', 'n']:
        for obj_type in ['V', 'E']:
            for block in (func_vals[:-1] if obj_type == 'E' else func_vals):

                # create lp variables for phi
                n_rows = myAssgn.phi(thickening, obj_type)[block].get_array().shape[0]
                n_cols = myAssgn.phi(thickening, obj_type)[block].get_array().shape[1]

                phi_vars[block][thickening][obj_type] = pulp.LpVariable.dicts('phi_'+thickening+obj_type+'_'+str(block), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

                # # set the initial values
                # for a in range(n_rows):
                #     for b in range(n_cols):
                #         prob += phi_vars[block][thickening][obj_type][(a,b)] == myAssgn.phi(thickening, obj_type)[block].get_array()[a][b]


                # create lp variables for psi
                n_rows = myAssgn.psi(thickening, obj_type)[block].get_array().shape[0]
                n_cols = myAssgn.psi(thickening, obj_type)[block].get_array().shape[1]

                psi_vars[block][thickening][obj_type] = pulp.LpVariable.dicts('psi_'+thickening+obj_type+'_'+str(block), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

                # # set the initial values
                # for a in range(n_rows):
                #     for b in range(n_cols):
                #         prob += psi_vars[block][thickening][obj_type][(a,b)] == myAssgn.psi(thickening, obj_type)[block].get_array()[a][b]

    # create the other decision variables
    z_vars = {block :  {starting_map: {obj_type: {} for obj_type in ['V', 'E']} for starting_map in ['F', 'G']} for block in func_vals}

    map_product_vars = {block : {starting_map: {obj_type: {} for obj_type in ['V', 'E']} for starting_map in ['F', 'G']} for block in func_vals}
    for obj_type in ['V', 'E']:
        for starting_map in ['F', 'G']:
            for block in (func_vals[:-1] if obj_type == 'E' else func_vals):
                
                if starting_map == 'F':
                    n_rows_1 = myAssgn.psi('n', obj_type)[block].get_array().shape[0]
                    n_cols_1 = myAssgn.phi('0', obj_type)[block].get_array().shape[1]
                    
                    # set the map product variables
                    map_product_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts(starting_map+'_'+obj_type+'_'+str(block), ((a, b) for a in range(n_rows_1) for b in range(n_cols_1)), cat='Integer')

                    n_rowcol_1 = myAssgn.psi('n', obj_type)[block].get_array().shape[1]

                    # set the z variables
                    z_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts('z_'+starting_map+'_'+obj_type+'_'+str(block), ((a, b, c) for a in range(n_rows_1) for b in range(n_rowcol_1) for c in range(n_cols_1)), cat='Binary')
                    
                else:
                    n_rows_1 = myAssgn.phi('n', obj_type)[block].get_array().shape[0]
                    n_cols_1 = myAssgn.psi('0', obj_type)[block].get_array().shape[1]

                    # set the map product variables
                    map_product_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts(starting_map+'_'+obj_type+'_'+str(block), ((a, b) for a in range(n_rows_1) for b in range(n_cols_1)), cat='Integer')

                    n_rowcol_2 = myAssgn.phi('n', obj_type)[block].get_array().shape[1]

                    z_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts('z_'+starting_map+'_'+obj_type+'_'+str(block), ((a, b, c) for a in range(n_rows_1) for b in range(n_rowcol_2) for c in range(n_cols_1)), cat='Binary')

    # decison variables for the triangles (to make the ceiling(expression/2 work)
    aux_vars = {block : {starting_map: {obj_type: {} for obj_type in ['V', 'E']} for starting_map in ['F', 'G']} for block in func_vals}

    for obj_type in ['V', 'E']:
        for starting_map in ['F', 'G']:
            # for block in func_vals:
            for block in (func_vals[:-1] if obj_type == 'E' else func_vals):
                if starting_map == 'F':
                    shape_m_tri = myAssgn.D('F', '2n', obj_type)[block].get_array().shape[0]
                else:
                    shape_m_tri = myAssgn.D('G', '2n', obj_type)[block].get_array().shape[0]

                aux_vars[block][starting_map][obj_type] = pulp.LpVariable('aux_'+starting_map+'_'+obj_type+'_'+str(block),  cat='Integer')


                    
    # create the minmax variable
    minmax_var = pulp.LpVariable('minmax_var', cat='Integer')

    
    # create the constraints
    for block in func_vals:
        for starting_map in ['F', 'G']:
            # set the other map based on starting map 
            if starting_map == 'F':
                other_map = 'G'
            else:
                other_map = 'F'

            for up_or_down in ['up', 'down']: # deals with 1 (up, down) and 2 (up, down)
                
                if block == func_vals[-1]: # skip the last block for this type of diagrams
                    continue
                
               #set the matrices
                if up_or_down == 'up': #NOTE: the change in block indices
                    dist_n_other = myAssgn.D(other_map, 'n', 'V')[block+1].get_array()
                    bou_n = myAssgn.B_up(other_map, 'n')[block].get_array()
                    bou_0 = myAssgn.B_up(starting_map, '0')[block].get_array()
                    if starting_map == 'F':
                        map_V = myAssgn.phi('0', 'V')[block+1].get_array()
                        map_E = myAssgn.phi('0', 'E')[block].get_array()
                        map_V_vars = phi_vars[block+1]['0']['V']
                        map_E_vars = phi_vars[block]['0']['E']
                    else:
                        map_V = myAssgn.psi('0', 'V')[block+1].get_array()
                        map_E = myAssgn.psi('0', 'E')[block].get_array()
                        map_V_vars = psi_vars[block+1]['0']['V']
                        map_E_vars = psi_vars[block]['0']['E']
                else:
                    dist_n_other = myAssgn.D(other_map, 'n', 'V')[block].get_array()
                    bou_n = myAssgn.B_down(other_map, 'n')[block].get_array()
                    bou_0 = myAssgn.B_down(starting_map, '0')[block].get_array()
                    if starting_map == 'F':
                        map_V = myAssgn.phi('0', 'V')[block].get_array()
                        map_E = myAssgn.phi('0', 'E')[block].get_array()
                        map_V_vars = phi_vars[block]['0']['V']
                        map_E_vars = phi_vars[block]['0']['E']
                    else:
                        map_V = myAssgn.psi('0', 'V')[block].get_array()
                        map_E = myAssgn.psi('0', 'E')[block].get_array()
                        map_V_vars = psi_vars[block]['0']['V']
                        map_E_vars = psi_vars[block]['0']['E']

                # set the dimensions
                shape_m_mix = dist_n_other.shape[0]
                shape_n_mix = map_V.shape[1]
                shape_o_mix = bou_n.shape[1]
                shape_p_mix = map_E.shape[1]

                # constraint 1: loss is bigger than the absolute value of each matrix elements

                for i in range(shape_m_mix):
                    for k in range(shape_p_mix):
                        # inner difference
                        first_term = pulp.lpSum([map_V_vars[i,j] * bou_0[j][k] for j in range(shape_n_mix)])
                        second_term = pulp.lpSum([bou_n[i][l] * map_E_vars[l,k] for l in range(shape_o_mix)])

                        # total expression
                        mixed_expression = pulp.lpSum(dist_n_other[i][h] * (first_term - second_term) for h in range(shape_m_mix))
                        
                        prob += minmax_var >= mixed_expression

                # constraint 2: each column sums to 1
                for j in range(shape_n_mix):
                    prob += pulp.lpSum(map_V_vars[h,j] for h in range(shape_m_mix)) == 1   

                for k in range(shape_p_mix):
                    prob += pulp.lpSum(map_E_vars[l, k] for l in range(shape_o_mix)) == 1



            for obj_type in ['V', 'E']: # deals with 3, 4, 5, 6, 7, 8, 9, 10
                if obj_type == 'E' and block == func_vals[-1]: # skip the last block for this type of diagrams. This is because we don't have an edge with the highest function value
                    continue

                # multiply inclusion matrices. Needed for the triangles
                i_n_i_0 = myAssgn.I(starting_map, 'n', obj_type)[block].get_array() @ myAssgn.I(starting_map, '0', obj_type)[block].get_array()

                # write inclusion matrices for easier reference. Needed for the parallelograms
                inc_0_para = myAssgn.I(starting_map, '0', obj_type)[block].get_array()
                inc_n_para = myAssgn.I(other_map, 'n', obj_type)[block].get_array()

                # write dist matrix for easier reference.
                # for triangles
                dist_2n_starting = myAssgn.D(starting_map, '2n', obj_type)[block].get_array()
                # for parallelograms
                dist_2n_other = myAssgn.D(other_map, '2n', obj_type)[block].get_array()
                
                # set map matrices for easier reference
                if starting_map == 'F':
                    map_0_para_vars = phi_vars[block]['0'][obj_type]
                    map_n_para_vars = phi_vars[block]['n'][obj_type]

                if starting_map == 'G':
                    map_0_para_vars = psi_vars[block]['0'][obj_type]
                    map_n_para_vars = psi_vars[block]['n'][obj_type]

                # set the dimensions
                shape_m_tri = dist_2n_starting.shape[0] # for triangles
                shape_m_para = dist_2n_other.shape[0] # for parallelograms

                shape_o_para = myAssgn.I(other_map, 'n', obj_type)[block].get_array().shape[1] # for parallelograms


                if starting_map == 'F':
                    shape_n_tri = myAssgn.psi('n', obj_type)[block].get_array().shape[1] # for triangles
                    shape_o_tri = myAssgn.phi('0', obj_type)[block].get_array().shape[1] # for triangles

                    shape_n_para = myAssgn.phi('n', obj_type)[block].get_array().shape[1] # for parallelograms
                    
                    shape_p_para = myAssgn.phi('0', obj_type)[block].get_array().shape[1] # for parallelograms
                else:
                    shape_n_tri = myAssgn.phi('n', obj_type)[block].get_array().shape[1] # for triangles
                    shape_o_tri = myAssgn.psi('0', obj_type)[block].get_array().shape[1] # for triangles

                    shape_n_para = myAssgn.psi('n', obj_type)[block].get_array().shape[1] # for parallelograms
                    shape_p_para = myAssgn.psi('0', obj_type)[block].get_array().shape[1] # for parallelograms


                


                #  constraint 1: loss is bigger than the absolute value of each matrix elements
            
                # for triangles
                for  h in range(shape_m_tri):                    

                    tri_expression = pulp.lpSum(dist_2n_starting[i,h] * (i_n_i_0[h,k] - map_product_vars[block][starting_map][obj_type][h,k]) for i in range(shape_m_tri) for k in range(shape_o_tri))

                    prob += aux_vars[block][starting_map][obj_type] * 2 >= tri_expression # ceiling of half of the expression

                    prob += minmax_var >= aux_vars[block][starting_map][obj_type]



                # for parallelograms
                for i in range(shape_m_para):
                    for k in range(shape_p_para):
                        # inner difference
                            first_term = pulp.lpSum([map_n_para_vars[i,j] * inc_0_para[j][k] for j in range(shape_n_para)])
                            second_term = pulp.lpSum([inc_n_para[i][l]  * map_0_para_vars[l,k] for l in range(shape_o_para)])


                            # total expression
                            para_expression = pulp.lpSum(dist_2n_other[i][h] * (first_term - second_term) for h in range(shape_m_para))

                            prob += minmax_var >= para_expression


                # constraint 2: map_multiplication and z relation. This is for triangles
                for i in range(shape_m_tri):
                    for k in range(shape_o_tri):
                        prob += map_product_vars[block][starting_map][obj_type][i,k] == pulp.lpSum(z_vars[block][starting_map][obj_type][i,j,k] for j in range(shape_n_tri))

                # constraint 3: z is less than either of the map values and greater than sum of the map values minus 1. This is for triangles
                for i in range(shape_m_tri):
                    for j in range(shape_n_tri):
                        for k in range(shape_o_tri):
                            if starting_map == 'F':
                                prob += z_vars[block][starting_map][obj_type][i,j,k] <= psi_vars[block]['n'][obj_type][i,j]
                                prob += z_vars[block][starting_map][obj_type][i,j,k] <= phi_vars[block]['0'][obj_type][j,k]
                                prob += z_vars[block][starting_map][obj_type][i,j,k] >= psi_vars[block]['n'][obj_type][i,j] + phi_vars[block]['0'][obj_type][j,k] - 1
                            else:
                                prob += z_vars[block][starting_map][obj_type][i,j,k] <= phi_vars[block]['n'][obj_type][i,j]
                                prob += z_vars[block][starting_map][obj_type][i,j,k] <= psi_vars[block]['0'][obj_type][j,k]
                                prob += z_vars[block][starting_map][obj_type][i,j,k] >= phi_vars[block]['n'][obj_type][i,j] + psi_vars[block]['0'][obj_type][j,k] - 1

                # constraint 4: each column sums to 1
                if starting_map == 'F':
                    # for triangles
                    for j in range(shape_n_tri):
                        prob += pulp.lpSum(psi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_tri)) == 1                        
                    for k in range(shape_o_tri):
                        prob += pulp.lpSum(phi_vars[block]['0'][obj_type][j,k] for j in range(shape_n_tri)) == 1

                    # for parallelograms
                    for j in range(shape_n_para):
                        prob += pulp.lpSum(phi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_para)) == 1

                    for k in range(shape_p_para):
                        prob += pulp.lpSum(phi_vars[block]['0'][obj_type][j,k] for j in range(shape_o_para)) == 1

                else:
                    # for triangles
                    for j in range(shape_n_tri):
                        prob += pulp.lpSum(phi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_tri)) == 1

                    for k in range(shape_o_tri):
                        prob += pulp.lpSum(psi_vars[block]['0'][obj_type][j,k] for j in range(shape_n_tri)) == 1

                    # for parallelograms
                    for j in range(shape_n_para):
                        prob += pulp.lpSum(psi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_para)) == 1

                    for k in range(shape_p_para):
                        prob += pulp.lpSum(psi_vars[block]['0'][obj_type][j,k] for j in range(shape_o_para)) == 1

    
    # Set the objective function
    prob += minmax_var

    # solve the problem
    if pulp_solver == 'GUROBI':   
        prob.solve(pulp.GUROBI_CMD(msg=0))
    else:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

    
    # prob.solve(pulp.GUROBI_CMD(msg=0))
    if prob.status != 1:
        raise ValueError("The ILP optimization did not converge. Please check the input data and try again.")

    # create a dictionary to store the results
    map_results = {'phi_vars': phi_vars, 'psi_vars': psi_vars}

    # make the results a LabeledBlockMatrix
    final_maps = build_map_matrices(myAssgn, map_results)
        
    if verbose:
        print(f"The optimized loss is: {pulp.value(minmax_var)}")
        print("Status:", pulp.LpStatus[prob.status])
        prob.writeLP("model.lp")  # Write the model in LP format
    
    # if get_thickened_maps:
    #     final_maps = build_map_matrices(myAssgn, map_results, thickening = 'n')

    #     return final_maps, pulp.value(minmax_var)

    # return results
    return final_maps, pulp.value(minmax_var)
