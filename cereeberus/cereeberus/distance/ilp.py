from cereeberus import ReebGraph, MapperGraph, Interleave
from .labeled_blocks import LabeledBlockMatrix as LBM
from .labeled_blocks import LabeledMatrix as LM
import cereeberus.data.ex_mappergraphs as ex_mg

import matplotlib.pyplot as plt
import numpy as np

import pulp #for ILP optimization



# function to build the phi and psi matrices after the ILP optimization
def build_map_matrices(myInt, map_results):
    """
    Function to build the map matrices after the ILP optimization.
    
    Parameters:
        myInt (Interleave): the Interleave object
        map_results (dict): the dictionary containing the results of the ILP optimization

    Returns:
        dict: the dictionary containing the final map matrices
    """

    # get the function values
    func_vals = myInt.all_func_vals()


    # create empty dictionaries to store the final map matrices
    final_maps = {block : {thickening:{obj_type:{} for obj_type in ['V', 'E']} for thickening in ['0', 'n']} for block in func_vals}

    # create the final map matrices
    
    final_maps = {thickening: {obj_type: {map_type: {} for map_type in ['Phi', 'Psi']} for obj_type in ['V', 'E']} for thickening in ['0', 'n']}
    #final_maps = {obj_type: {map_type:{} for map_type in ['Phi','Psi']}  for obj_type in ['V', 'E']}
    for thickening in ['0', 'n']:
        for obj_type in ['V', 'E']:
            for map_type in ['Phi', 'Psi']:
                
                for block in func_vals:
                    if map_type == 'Phi':
                        map_vars = myInt.phi(thickening, obj_type)[block].get_array()
                        block_row_labels = myInt.phi(thickening, obj_type)[block].rows
                        block_col_labels = myInt.phi(thickening, obj_type)[block].cols
                    else:
                        map_vars = myInt.psi(thickening, obj_type)[block].get_array()
                        block_row_labels = myInt.psi(thickening, obj_type)[block].rows
                        block_col_labels = myInt.psi(thickening, obj_type)[block].cols


                    n_rows = map_vars.shape[0]
                    n_cols = map_vars.shape[1]

                    # create the final map matrix
                    final_block_map = np.zeros((n_rows, n_cols))

                    # set the values
                    for a in range(n_rows):
                        for b in range(n_cols):
                            final_block_map[a,b] = pulp.value(map_results[map_type+'_vars'][block][thickening][obj_type][(a,b)])

                    # make a LabeledMatrix
                    final_block_map = LM(final_block_map, block_row_labels, block_col_labels)

                    
                    # store the final map matrix
                    final_maps[thickening][obj_type][map_type][block] = final_block_map

    # make labeled block matrices
    Phi_0_V_LBM = LBM(labled_matrix_dict=final_maps['0']['V']['Phi'])
    Phi_0_E_LBM = LBM(labled_matrix_dict=final_maps['0']['E']['Phi'])
    Phi_n_V_LBM = LBM(labled_matrix_dict=final_maps['n']['V']['Phi'])
    Phi_n_E_LBM = LBM(labled_matrix_dict=final_maps['n']['E']['Phi'])
    Psi_0_V_LBM = LBM(labled_matrix_dict=final_maps['0']['V']['Psi'])
    Psi_0_E_LBM = LBM(labled_matrix_dict=final_maps['0']['E']['Psi'])
    Psi_n_V_LBM = LBM(labled_matrix_dict=final_maps['n']['V']['Psi'])
    Psi_n_E_LBM = LBM(labled_matrix_dict=final_maps['n']['E']['Psi'])

    final_LBMs = {'Phi_0_V': Phi_0_V_LBM, 'Phi_0_E': Phi_0_E_LBM, 'Phi_n_V': Phi_n_V_LBM, 'Phi_n_E': Phi_n_E_LBM, 'Psi_0_V': Psi_0_V_LBM, 'Psi_0_E': Psi_0_E_LBM, 'Psi_n_V': Psi_n_V_LBM, 'Psi_n_E': Psi_n_E_LBM}



    return final_LBMs
    


    
##------------------- ILP Optimization -------------------##


def solve_ilp(myInt, verbose=False, get_thickened_maps = False):

    # function values
    func_vals = myInt.all_func_vals()

    # create the ILP problem
    prob = pulp.LpProblem("Interleave_Optimization_Problem", pulp.LpMinimize)

    # create empty dictionaries to store the decision variables
    Phi_vars = {block : {thickening:{obj_type:{} for obj_type in ['V', 'E']} for thickening in ['0', 'n']} for block in func_vals}
    Psi_vars = {block : {thickening:{obj_type:{} for obj_type in ['V', 'E']} for thickening in ['0', 'n']} for block in func_vals}


    # create the decision variables (NOTE: these are all the decision variables for all the diagrams)
    for block in func_vals:
        for thickening in ['0', 'n']:
            for obj_type in ['V', 'E']:
                
                # create lp variables for phi
                n_rows = myInt.phi(thickening, obj_type)[block].get_array().shape[0]
                n_cols = myInt.phi(thickening, obj_type)[block].get_array().shape[1]

                Phi_vars[block][thickening][obj_type] = pulp.LpVariable.dicts('Phi_'+thickening+obj_type+'_'+str(block), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

                # # set the initial values
                # for a in range(n_rows):
                #     for b in range(n_cols):
                #         prob += Phi_vars[block][thickening][obj_type][(a,b)] == myInt.phi(thickening, obj_type)[block].get_array()[a][b]


                # create lp variables for psi
                n_rows = myInt.psi(thickening, obj_type)[block].get_array().shape[0]
                n_cols = myInt.psi(thickening, obj_type)[block].get_array().shape[1]

                Psi_vars[block][thickening][obj_type] = pulp.LpVariable.dicts('Psi_'+thickening+obj_type+'_'+str(block), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

                # # set the initial values
                # for a in range(n_rows):
                #     for b in range(n_cols):
                #         prob += Psi_vars[block][thickening][obj_type][(a,b)] == myInt.psi(thickening, obj_type)[block].get_array()[a][b]

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

                    # set the z variables
                    z_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts('z_'+starting_map+'_'+obj_type+'_'+str(block), ((a, b, c) for a in range(n_rows_1) for b in range(n_rowcol_1) for c in range(n_cols_1)), cat='Binary')
                    
                else:
                    n_rows_1 = myInt.phi('n', obj_type)[block].get_array().shape[0]
                    n_cols_1 = myInt.psi('0', obj_type)[block].get_array().shape[1]

                    # set the map product variables
                    map_product_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts(starting_map+'_'+obj_type+'_'+str(block), ((a, b) for a in range(n_rows_1) for b in range(n_cols_1)), cat='Integer')

                    n_rowcol_2 = myInt.phi('n', obj_type)[block].get_array().shape[1]

                    z_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts('z_'+starting_map+'_'+obj_type+'_'+str(block), ((a, b, c) for a in range(n_rows_1) for b in range(n_rowcol_2) for c in range(n_cols_1)), cat='Binary')

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
               
               #set the matrices
                if up_or_down == 'up': #NOTE: the change in block indices

                    if block == func_vals[-1]: # skip the last block for the up case
                        continue
                    
                    dist_n_other = myInt.D(other_map, 'n', 'V')[block+1].get_array()
                    bou_n = myInt.B_up(other_map, 'n')[block].get_array()
                    bou_0 = myInt.B_up(starting_map, '0')[block].get_array()
                    if starting_map == 'F':
                        map_V = myInt.phi('0', 'V')[block+1].get_array()
                        map_E = myInt.phi('0', 'E')[block].get_array()
                        map_V_vars = Phi_vars[block+1]['0']['V']
                        map_E_vars = Phi_vars[block]['0']['E']
                    else:
                        map_V = myInt.psi('0', 'V')[block+1].get_array()
                        map_E = myInt.psi('0', 'E')[block].get_array()
                        map_V_vars = Psi_vars[block+1]['0']['V']
                        map_E_vars = Psi_vars[block]['0']['E']
                else:
                    dist_n_other = myInt.D(other_map, 'n', 'V')[block].get_array()
                    bou_n = myInt.B_down(other_map, 'n')[block].get_array()
                    bou_0 = myInt.B_down(starting_map, '0')[block].get_array()
                    if starting_map == 'F':
                        map_V = myInt.phi('0', 'V')[block].get_array()
                        map_E = myInt.phi('0', 'E')[block].get_array()
                        map_V_vars = Phi_vars[block]['0']['V']
                        map_E_vars = Phi_vars[block]['0']['E']
                    else:
                        map_V = myInt.psi('0', 'V')[block].get_array()
                        map_E = myInt.psi('0', 'E')[block].get_array()
                        map_V_vars = Psi_vars[block]['0']['V']
                        map_E_vars = Psi_vars[block]['0']['E']

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
                        expression = pulp.lpSum(dist_n_other[i][h] * (first_term - second_term) for h in range(shape_m_mix))
                        
                        prob += minmax_var >= expression
                        prob += - minmax_var <= expression

                # constraint 2: each column sums to 1
                for j in range(shape_n_mix):
                    prob += pulp.lpSum(map_V_vars[h,j] for h in range(shape_m_mix)) == 1   

                for k in range(shape_p_mix):
                    prob += pulp.lpSum(map_E_vars[l, k] for l in range(shape_o_mix)) == 1



            for obj_type in ['V', 'E']: # deals with 3, 4, 5, 6, 7, 8, 9, 10
                # multiply inclusion matrices. Needed for the triangles
                i_n_i_0 = myInt.I(starting_map, 'n', obj_type)[block].get_array() @ myInt.I(starting_map, '0', obj_type)[block].get_array()

                # write inclusion matrices for easier reference. Needed for the parallelograms
                inc_0_para = myInt.I(starting_map, '0', obj_type)[block].get_array()
                inc_n_para = myInt.I(other_map, 'n', obj_type)[block].get_array()

                # write dist matrix for easier reference.
                # for triangles
                dist_2n_starting = myInt.D(starting_map, '2n', obj_type)[block].get_array()
                # for parallelograms
                dist_2n_other = myInt.D(other_map, '2n', obj_type)[block].get_array()
                
                # set map matrices for easier reference
                if starting_map == 'F':
                    map_0_para_vars = Phi_vars[block]['0'][obj_type]
                    map_n_para_vars = Phi_vars[block]['n'][obj_type]

                if starting_map == 'G':
                    map_0_para_vars = Psi_vars[block]['0'][obj_type]
                    map_n_para_vars = Psi_vars[block]['n'][obj_type]

                # set the dimensions
                shape_m_tri = dist_2n_starting.shape[0] # for triangles
                shape_m_para = dist_2n_other.shape[0] # for parallelograms

                shape_o_para = myInt.I(other_map, 'n', obj_type)[block].get_array().shape[1] # for parallelograms


                if starting_map == 'F':
                    shape_n_tri = myInt.psi('n', obj_type)[block].get_array().shape[1] # for triangles
                    shape_o_tri = myInt.phi('0', obj_type)[block].get_array().shape[1] # for triangles

                    shape_n_para = myInt.phi('n', obj_type)[block].get_array().shape[1] # for parallelograms
                    
                    shape_p_para = myInt.phi('0', obj_type)[block].get_array().shape[1] # for parallelograms
                else:
                    shape_n_tri = myInt.phi('n', obj_type)[block].get_array().shape[1] # for triangles
                    shape_o_tri = myInt.psi('0', obj_type)[block].get_array().shape[1] # for triangles

                    shape_n_para = myInt.psi('n', obj_type)[block].get_array().shape[1] # for parallelograms
                    shape_p_para = myInt.psi('0', obj_type)[block].get_array().shape[1] # for parallelograms


                


                #  constraint 1: loss is bigger than the absolute value of each matrix elements
            
                # for triangles
                for  h in range(shape_m_tri):                    
                    prob += minmax_var >= pulp.lpSum(dist_2n_starting[i,h] * (i_n_i_0[h,k] - map_product_vars[block][starting_map][obj_type][h,k]) for i in range(shape_m_tri) for k in range(shape_o_tri))
                    prob += - minmax_var <= pulp.lpSum(dist_2n_starting[i,h] * (i_n_i_0[h,k] - map_product_vars[block][starting_map][obj_type][h,k]) for i in range(shape_m_tri) for k in range(shape_o_tri))

                # for parallelograms
                for i in range(shape_m_para):
                    for k in range(shape_p_para):
                        # inner difference
                            first_term = pulp.lpSum([map_n_para_vars[i,j] * inc_0_para[j][k] for j in range(shape_n_para)])
                            second_term = pulp.lpSum([inc_n_para[i][l]  * map_0_para_vars[l,k] for l in range(shape_o_para)])


                            # total expression
                            expression = pulp.lpSum(dist_2n_other[i][h] * (first_term - second_term) for h in range(shape_m_para))

                            prob += minmax_var >= expression
                            prob += - minmax_var <= expression


                # constraint 2: map_multiplication and z relation. This is for triangles
                for i in range(shape_m_tri):
                    for k in range(shape_o_tri):
                        prob += map_product_vars[block][starting_map][obj_type][i,k] == pulp.lpSum(z_vars[block][starting_map][obj_type][i,j,k] for j in range(shape_n_tri))

                # constraint 3: z is less than either of the map values and greater than sum of the map values minus 1. This is for triangles
                for i in range(shape_m_tri):
                    for j in range(shape_n_tri):
                        for k in range(shape_o_tri):
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
                    # for triangles
                    for j in range(shape_n_tri):
                        prob += pulp.lpSum(Psi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_tri)) == 1                        
                    for k in range(shape_o_tri):
                        prob += pulp.lpSum(Phi_vars[block]['0'][obj_type][j,k] for j in range(shape_n_tri)) == 1

                    # for parallelograms
                    for j in range(shape_n_para):
                        prob += pulp.lpSum(Phi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_para)) == 1

                    for k in range(shape_p_para):
                        prob += pulp.lpSum(Phi_vars[block]['0'][obj_type][j,k] for j in range(shape_o_para)) == 1

                else:
                    # for triangles
                    for j in range(shape_n_tri):
                        prob += pulp.lpSum(Phi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_tri)) == 1

                    for k in range(shape_o_tri):
                        prob += pulp.lpSum(Psi_vars[block]['0'][obj_type][j,k] for j in range(shape_n_tri)) == 1

                    # for parallelograms
                    for j in range(shape_n_para):
                        prob += pulp.lpSum(Psi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_para)) == 1

                    for k in range(shape_p_para):
                        prob += pulp.lpSum(Psi_vars[block]['0'][obj_type][j,k] for j in range(shape_o_para)) == 1

    
    # Set the objective function
    prob += minmax_var


    # solve the problem
    if verbose:
        prob.solve()
    else:
        prob.solve(pulp.GUROBI_CMD(msg=0)
                   )

    # create a dictionary to store the results
    map_results = {'Phi_vars': Phi_vars, 'Psi_vars': Psi_vars}


    # make the results a LabeledBlockMatrix
    final_maps = build_map_matrices(myInt, map_results)
        
    if verbose:
        print(f"The optimized loss is: {pulp.value(minmax_var)}")
        print("Status:", pulp.LpStatus[prob.status])
        prob.writeLP("model.lp")  # Write the model in LP format
    
    if get_thickened_maps:
        final_maps = build_map_matrices(myInt, map_results, thickening = 'n')

        return final_maps, pulp.value(minmax_var)

    # return results
    return final_maps, pulp.value(minmax_var)
