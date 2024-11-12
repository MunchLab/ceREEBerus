from cereeberus import ReebGraph, MapperGraph, Interleave
import cereeberus.data.ex_mappergraphs as ex_mg

import matplotlib.pyplot as plt
import numpy as np

import pulp #for ILP optimization



##------------------- ILP Optimization -------------------##


def solve_ilp(myInt):

    # function values
    func_vals = myInt.all_func_vals()

    # create the ILP problem
    prob = pulp.LpProblem("Interleave Optimization Problem", pulp.LpMinimize)

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
                        prob += -minmax_var <= expression

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
                    prob += -minmax_var <= pulp.lpSum(dist_2n_starting[i,h] * (i_n_i_0[h,k] - map_product_vars[block][starting_map][obj_type][h,k]) for i in range(shape_m_tri) for k in range(shape_o_tri))

                # for parallelograms
                for i in range(shape_m_para):
                    for k in range(shape_p_para):
                        # inner difference
                            first_term = pulp.lpSum([map_n_para_vars[i,j] * inc_0_para[j][k] for j in range(shape_n_para)])
                            second_term = pulp.lpSum([inc_n_para[i][l]  * map_0_para_vars[l,k] for l in range(shape_o_para)])


                            # total expression
                            expression = pulp.lpSum(dist_2n_other[i][h] * (first_term - second_term) for h in range(shape_m_para))

                            prob += minmax_var >= expression
                            prob += -minmax_var <= expression


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

                
                
    # solve the problem
    prob.solve()

    # print the loss
    print(f"Loss: {pulp.value(minmax_var)}")

    

    prob.writeLP("model.lp")  # Write the model in LP format






##------------------- Break it down to functions -------------------##

# # retrive all the function values of the mappers
# def set_function_values(myInt):
#     """
#     Set the function values for the ILP optimization problem.
    
#     Parameters
    
#         myInt: Interleave object
#         The assignment that we want to optimize.
        
        
#     Returns
#         list
#             The function values of the two mappers.
#     """
    
#     func_vals = [5]
#     return func_vals

# def create_map_variables(myInt, prob):
#     """
#     Create the decision variables for the ILP optimization problem. These are Phi, Phi^n, Psi, and Psi^n. Also set the initial values of the decision variables.
    
#     Parameters
    
#         myInt: Interleave object
#         The assignment that we want to optimize.
        
#         prob: pulp.LpProblem
#         The ILP optimization problem.
        
#     Returns
#         Dict
#             The decision variables (Phi, Phi^n, Psi, Psi^n)."""
    
#     # get the function values
#     func_vals = set_function_values(myInt)

#     # Initial maps
#     # phi
#     Phi = {
#     'Phi_0V': myInt.phi('0','V'),
#     'Phi_nV': myInt.phi('n','V'),
#     'Phi_0E': myInt.phi('0','E'),
#     'Phi_nE': myInt.phi('n','E')
#     }
#     # psi
#     Psi = {
#     'Psi_0V': myInt.psi('0','V'),
#     'Psi_nV': myInt.psi('n','V'),
#     'Psi_0E': myInt.psi('0','E'),
#     'Psi_nE': myInt.psi('n','E')
#     }

#     # create the decision variables at the block level
#     Phi_vars = {'V':{'0':{},'n':{}},'E':{'0':{},'n':{}}} # to store all the lp variables for phi
#     Psi_vars = {'V':{'0':{},'n':{}},'E':{'0':{},'n':{}}} # to store all the lp variables for psi

#     for i in func_vals:
#        for j in ['0', 'n']:
#               for k in ['V', 'E']:
#                     # phi

#                     # create variables for each block
#                     n_rows = Phi['Phi_'+j+k][i].get_array().shape[0]
#                     n_cols = Phi['Phi_'+j+k][i].get_array().shape[1]
#                     Phi_vars[k][j][i] = pulp.LpVariable.dicts('Phi_'+j+k+'_'+str(i), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

#                     # set the initial values
#                     for a in range(n_rows):
#                         for b in range(n_cols):
#                              prob += Phi_vars[k][j][i][(a,b)] == Phi['Phi_'+j+k][i].get_array()[a][b]

#                     # psi
#                     n_rows = Psi['Psi_'+j+k][i].get_array().shape[0]
#                     n_cols = Psi['Psi_'+j+k][i].get_array().shape[1]
#                     Psi_vars[k][j][i] = pulp.LpVariable.dicts('Psi_'+j+k+'_'+str(i), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

#                     # set the initial values
#                     for a in range(n_rows):
#                         for b in range(n_cols):
#                              prob += Psi_vars[k][j][i][(a,b)] == Psi['Psi_'+j+k][i].get_array()[a][b]

#     return {'Phi_vars':Phi_vars, 'Psi_vars':Psi_vars}

# def create_other_decision_variables(myInt):
#     """
#     Create the other decision variables for the ILP optimization problem. These are the reparametrization variables.

#     Parameters

#         myInt: Interleave object
#         The assignment that we want to optimize.
        
#         prob: pulp.LpProblem
#         The ILP optimization problem.
        
#     Returns
#         Dict
#             The decision variables z, map_product."""

#     # get the function values
#     func_vals = set_function_values(myInt)

#     # the map matrices
#     Phi = {
#     'Phi_0V': myInt.phi('0','V'),
#     'Phi_nV': myInt.phi('n','V'),
#     'Phi_0E': myInt.phi('0','E'),
#     'Phi_nE': myInt.phi('n','E')
#     }
#     Psi = {
#     'Psi_0V': myInt.psi('0','V'),
#     'Psi_nV': myInt.psi('n','V'),
#     'Psi_0E': myInt.psi('0','E'),
#     'Psi_nE': myInt.psi('n','E')
#     }

#     # create the decision variables at the block level
#     z_vars = {'V':{'FGF':{}, 'GFG':{}},'E':{'FGF':{}, 'GFG':{}}}
#     map_product_vars = {'V':{'FGF':{}, 'GFG':{}},'E':{'FGF':{}, 'GFG':{}}}

#     for i in func_vals:
#         for k in ['V', 'E']:
#             n_rows_1 = Psi['Psi_n'+k][i].get_array().shape[0]
#             n_cols_1 = Phi['Phi_0'+k][i].get_array().shape[1]

#             map_product_vars[k]['FGF'][i] = pulp.LpVariable.dicts('FGF_'+str(i), ((a, b) for a in range(n_rows_1) for b in range(n_cols_1)), cat='Integer')

#             n_rows_2 = Phi['Phi_n'+k][i].get_array().shape[0]
#             n_cols_2 = Psi['Psi_0'+k][i].get_array().shape[1]

#             map_product_vars[k]['GFG'][i] = pulp.LpVariable.dicts('GFG_'+str(i), ((a, b) for a in range(n_rows_2) for b in range(n_cols_2)), cat='Integer')

#             n_rowcol_1 = Psi['Psi_n'+k][i].get_array().shape[1]
#             n_rowcol_2 = Phi['Phi_n'+k][i].get_array().shape[1]

#             z_vars[k]['FGF'][i] = pulp.LpVariable.dicts('z_FGF_'+str(i), ((a, b, c) for a in range(n_rows_1) for b in range(n_rowcol_1) for c in range(n_cols_1)), cat='Binary')

#             z_vars[k]['GFG'][i] = pulp.LpVariable.dicts('z_GFG_'+str(i), ((a, b, c) for a in range(n_rows_2) for b in range(n_rowcol_2) for c in range(n_cols_2)), cat='Binary')

#     return {'z':z_vars, 'map_product':map_product_vars}


# def set_objective_function(prob):
#     """
#     Set the objective function for the ILP optimization problem. This is created across all the blocks.
    
#     Parameters
    
#         prob: pulp.LpProblem
#         The ILP optimization problem.
        
#     Returns
#         pulp.LpVariable
#             The minmax variable."""
    
#     # create the minmax variable
#     minmax_var = pulp.LpVariable('minmax_var', lowBound=0, cat='Continuous')

    
#     # define the objective function
#     prob += minmax_var

#     return minmax_var  


# def set_triangle_constraints(myInt, prob):
#     """
#     Set the triangle constraints for the ILP optimization problem. These constraints are created block by block.
    
#     Parameters
    
#         myInt: Interleave object
#         The assignment that we want to optimize.
        
#         prob: pulp.LpProblem
#         The ILP optimization problem.
        
#     Returns
#         None"""
    
#     # get the function values
#     func_vals = set_function_values(myInt)

#     Phi = {
#     'Phi_0V': myInt.phi('0','V'),
#     'Phi_nV': myInt.phi('n','V'),
#     'Phi_0E': myInt.phi('0','E'),
#     'Phi_nE': myInt.phi('n','E')
#     }
#     Psi = {
#     'Psi_0V': myInt.psi('0','V'),
#     'Psi_nV': myInt.psi('n','V'),
#     'Psi_0E': myInt.psi('0','E'),
#     'Psi_nE': myInt.psi('n','E')
#     }

#     I = {
#     # vert
#     'I_F0V': myInt.I('F','0','V'),
#     'I_FnV': myInt.I('F','n','V'),
#     'I_G0V': myInt.I('G','0','V'),
#     'I_GnV': myInt.I('G','n','V'),

#     # edge
#     'I_F0E': myInt.I('F','0','E'),
#     'I_FnE': myInt.I('F','n','E'),
#     'I_G0E': myInt.I('G','0','E'),
#     'I_GnE': myInt.I('G','n','E')
#     }
    

#     D = {
#     # vert
#     'D_F2nV': myInt.D('F', '2n', 'V'),
#     'D_G2nV': myInt.D('G', '2n', 'V'),

#     # edge
#     'D_F2nE': myInt.D('F', '2n', 'E'),
#     'D_G2nE': myInt.D('G', '2n', 'E')
#     }

#     # phi
#     Phi = {
#     'Phi_0V': myInt.phi('0','V'),
#     'Phi_nV': myInt.phi('n','V'),
#     'Phi_0E': myInt.phi('0','E'),
#     'Phi_nE': myInt.phi('n','E')
#     }
#     # psi
#     Psi = {
#     'Psi_0V': myInt.psi('0','V'),
#     'Psi_nV': myInt.psi('n','V'),
#     'Psi_0E': myInt.psi('0','E'),
#     'Psi_nE': myInt.psi('n','E')
#     }

#     # map variables
#     Phi_vars = create_map_variables(myInt, prob)['Phi_vars']
#     Psi_vars = create_map_variables(myInt, prob)['Psi_vars']

#     # other variables
#     z_vars = create_other_decision_variables(myInt)['z']
#     map_product_vars = create_other_decision_variables(myInt)['map_product']

#     # minmax variable
#     minmax_var = set_objective_function(prob)

#     for i in func_vals:
#         print(f"iteration {i}")
#         for j in ['V', 'E']:
#             for k in ['F', 'G']:
#                 # multiply inclusion matrices
#                 i_n_i_0 = I['I_'+k+'n'+j][i].get_array() @ I['I_'+k+'0'+j][i].get_array()
                

#                 # set maps
#                 if k == 'F':
#                     map_1 = Psi['Psi_n'+j][i]
#                     map_2 = Phi['Phi_0'+j][i]
#                     map_1_vars = Psi_vars[j]['n'][i]
#                     map_2_vars = Phi_vars[j]['0'][i]
#                     map_product_var = map_product_vars[j]['FGF'][i]
#                     z_var = z_vars[j]['FGF'][i]
#                 else:  
#                     map_1 = Phi['Phi_n'+j][i]
#                     map_2 = Psi['Psi_0'+j][i]
#                     map_1_vars = Phi_vars[j]['n'][i]
#                     map_2_vars = Psi_vars[j]['0'][i]
#                     map_product_var = map_product_vars[j]['GFG'][i]
#                     z_var = z_vars[j]['GFG'][i]
             
#                 # write d
#                 dist = D['D_'+k+'2n'+j][i].get_array()
#                 # set the dimensions 
#                 shape_m_tri = dist.shape[0]
#                 shape_n_tri = map_1.get_array().shape[1]
#                 shape_o_tri = map_2.get_array().shape[1]

                
#                 #  constraint 1: loss is bigger than the absolute value of each matrix elements
#                 for a in range(shape_m_tri):
#                     prob += minmax_var >= pulp.lpSum(dist[b, a] * (i_n_i_0[a, d] - map_product_var[a, d]) for b in range(shape_m_tri) for d in range(shape_o_tri))
#                     prob += -minmax_var <= pulp.lpSum(dist[b, a] * (i_n_i_0[a, d] - map_product_var[a, d]) for b in range(shape_m_tri) for d in range(shape_o_tri))

                

#                 # constraints 2: map_multiplication and z relation
#                 for a in range(shape_m_tri):
#                     for d in range(shape_o_tri):
#                         prob += map_product_var[a, d] == pulp.lpSum(z_var[a, c, d] for c in range(shape_n_tri))


#                 # constraint 3: z is less than either of the map values and greater than sum of the map values minus 1
#                 for a in range(shape_m_tri):
#                     for c in range(shape_n_tri):
#                         for d in range(shape_o_tri):
#                             prob += z_var[a, c, d] <= map_1_vars[a, c]
#                             prob += z_var[a, c, d] <= map_2_vars[c, d]
#                             prob += z_var[a, c, d] >= map_1_vars[a, c] + map_2_vars[c, d] - 1

#                 # constraint 4: each column sums to 1
#                 for c in range(shape_n_tri):
#                     prob += pulp.lpSum(map_1_vars[a, c] for a in range(shape_m_tri)) == 1

#                 for d in range(shape_o_tri):
#                     prob += pulp.lpSum(map_2_vars[c, d] for c in range(shape_n_tri)) == 1



# def create_ilp_problem(myInt):
#     """
#     Create the ILP optimization problem.
    
#     Parameters
    
#         myInt: Interleave object
#         The assignment that we want to optimize.
        
#     Returns
    
#         prob: pulp.LpProblem
#         The ILP optimization problem.
#     """
    
#     prob = pulp.LpProblem("Interleave Optimization Problem", pulp.LpMinimize)
    
#     # Create and initialize decision matrices
#     create_map_variables(myInt, prob)

#     # create other decision variables
#     create_other_decision_variables(myInt)
    
#     # Set the objective function
#     set_objective_function(prob)
    
#     # Set the constraints
#     set_triangle_constraints(myInt, prob)

#     from cereeberus import ReebGraph, MapperGraph, Interleave
# import cereeberus.data.ex_mappergraphs as ex_mg

# import matplotlib.pyplot as plt
# import numpy as np

# import pulp #for ILP optimization

# # retrive all the function values of the mappers
# def set_function_values(myInt):
#     """
#     Set the function values for the ILP optimization problem.
    
#     Parameters
    
#         myInt: Interleave object
#         The assignment that we want to optimize.
        
        
#     Returns
#         list
#             The function values of the two mappers.
#     """
    
#     func_vals = myInt.all_func_vals()

#     return func_vals

# def create_map_variables(myInt, prob):
#     """
#     Create the decision variables for the ILP optimization problem. These are Phi, Phi^n, Psi, and Psi^n. Also set the initial values of the decision variables.
    
#     Parameters
    
#         myInt: Interleave object
#         The assignment that we want to optimize.
        
#         prob: pulp.LpProblem
#         The ILP optimization problem.
        
#     Returns
#         Dict
#             The decision variables (Phi, Phi^n, Psi, Psi^n)."""
    
#     # get the function values
#     func_vals = set_function_values(myInt)

#     # Initial maps
#     # phi
#     Phi = {
#     'Phi_0V': myInt.phi('0','V'),
#     'Phi_nV': myInt.phi('n','V'),
#     'Phi_0E': myInt.phi('0','E'),
#     'Phi_nE': myInt.phi('n','E')
#     }
#     # psi
#     Psi = {
#     'Psi_0V': myInt.psi('0','V'),
#     'Psi_nV': myInt.psi('n','V'),
#     'Psi_0E': myInt.psi('0','E'),
#     'Psi_nE': myInt.psi('n','E')
#     }

#     # create the decision variables at the block level
#     Phi_vars = {'V':{'0':{},'n':{}},'E':{'0':{},'n':{}}} # to store all the lp variables for phi
#     Psi_vars = {'V':{'0':{},'n':{}},'E':{'0':{},'n':{}}} # to store all the lp variables for psi

#     for i in func_vals:
#        for j in ['0', 'n']:
#               for k in ['V', 'E']:
#                     # phi

#                     # create variables for each block
#                     n_rows = Phi['Phi_'+j+k][i].get_array().shape[0]
#                     n_cols = Phi['Phi_'+j+k][i].get_array().shape[1]
#                     Phi_vars[k][j][i] = pulp.LpVariable.dicts('Phi_'+j+k+'_'+str(i), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

#                     # set the initial values
#                     for a in range(n_rows):
#                         for b in range(n_cols):
#                              prob += Phi_vars[k][j][i][(a,b)] == Phi['Phi_'+j+k][i].get_array()[a][b]

#                     # psi
#                     n_rows = Psi['Psi_'+j+k][i].get_array().shape[0]
#                     n_cols = Psi['Psi_'+j+k][i].get_array().shape[1]
#                     Psi_vars[k][j][i] = pulp.LpVariable.dicts('Psi_'+j+k+'_'+str(i), ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary')

#                     # set the initial values
#                     for a in range(n_rows):
#                         for b in range(n_cols):
#                              prob += Psi_vars[k][j][i][(a,b)] == Psi['Psi_'+j+k][i].get_array()[a][b]

#     return {'Phi_vars':Phi_vars, 'Psi_vars':Psi_vars}

# def create_other_decision_variables(myInt):
#     """
#     Create the other decision variables for the ILP optimization problem. These are the reparametrization variables.

#     Parameters

#         myInt: Interleave object
#         The assignment that we want to optimize.
        
#         prob: pulp.LpProblem
#         The ILP optimization problem.
        
#     Returns
#         Dict
#             The decision variables z, map_product."""

#     # get the function values
#     func_vals = set_function_values(myInt)

#     # the initial map matrices
#     Phi = {
#     'Phi_0V': myInt.phi('0','V'),
#     'Phi_nV': myInt.phi('n','V'),
#     'Phi_0E': myInt.phi('0','E'),
#     'Phi_nE': myInt.phi('n','E')
#     }
#     Psi = {
#     'Psi_0V': myInt.psi('0','V'),
#     'Psi_nV': myInt.psi('n','V'),
#     'Psi_0E': myInt.psi('0','E'),
#     'Psi_nE': myInt.psi('n','E')
#     }

#     # create the decision variables at the block level
#     z_vars = {'V':{'FGF':{}, 'GFG':{}},'E':{'FGF':{}, 'GFG':{}}}
#     map_product_vars = {'V':{'FGF':{}, 'GFG':{}},'E':{'FGF':{}, 'GFG':{}}}

#     for i in func_vals:
#         for k in ['V', 'E']:
#             n_rows_1 = Psi['Psi_n'+k][i].get_array().shape[0]
#             n_cols_1 = Phi['Phi_0'+k][i].get_array().shape[1]

#             map_product_vars[k]['FGF'][i] = pulp.LpVariable.dicts('FGF_'+str(i), ((a, b) for a in range(n_rows_1) for b in range(n_cols_1)), cat='Integer')

#             n_rows_2 = Phi['Phi_n'+k][i].get_array().shape[0]
#             n_cols_2 = Psi['Psi_0'+k][i].get_array().shape[1]

#             map_product_vars[k]['GFG'][i] = pulp.LpVariable.dicts('GFG_'+str(i), ((a, b) for a in range(n_rows_2) for b in range(n_cols_2)), cat='Integer')

#             n_rowcol_1 = Psi['Psi_n'+k][i].get_array().shape[1]
#             n_rowcol_2 = Phi['Phi_n'+k][i].get_array().shape[1]

#             z_vars[k]['FGF'][i] = pulp.LpVariable.dicts('z_FGF_'+str(i), ((a, b, c) for a in range(n_rows_1) for b in range(n_rowcol_1) for c in range(n_cols_1)), cat='Binary')

#             z_vars[k]['GFG'][i] = pulp.LpVariable.dicts('z_GFG_'+str(i), ((a, b, c) for a in range(n_rows_2) for b in range(n_rowcol_2) for c in range(n_cols_2)), cat='Binary')

#     return {'z':z_vars, 'map_product':map_product_vars}


# def set_objective_function(prob):
#     """
#     Set the objective function for the ILP optimization problem. This is created across all the blocks.
    
#     Parameters
    
#         prob: pulp.LpProblem
#         The ILP optimization problem.
        
#     Returns
#         pulp.LpVariable
#             The minmax variable."""
    
#     # create the minmax variable
#     minmax_var = pulp.LpVariable('minmax_var', lowBound=0, cat='Continuous')

    
#     # define the objective function
#     prob += minmax_var

#     return minmax_var  


# def set_triangle_constraints(myInt, prob):
#     """
#     Set the triangle constraints for the ILP optimization problem. These constraints are created block by block.
    
#     Parameters
    
#         myInt: Interleave object
#         The assignment that we want to optimize.
        
#         prob: pulp.LpProblem
#         The ILP optimization problem.
        
#     Returns
#         None"""
    
#     # get the function values
#     func_vals = set_function_values(myInt)

#     Phi = {
#     'Phi_0V': myInt.phi('0','V'),
#     'Phi_nV': myInt.phi('n','V'),
#     'Phi_0E': myInt.phi('0','E'),
#     'Phi_nE': myInt.phi('n','E')
#     }
#     Psi = {
#     'Psi_0V': myInt.psi('0','V'),
#     'Psi_nV': myInt.psi('n','V'),
#     'Psi_0E': myInt.psi('0','E'),
#     'Psi_nE': myInt.psi('n','E')
#     }

#     I = {
#     # vert
#     'I_F0V': myInt.I('F','0','V'),
#     'I_FnV': myInt.I('F','n','V'),
#     'I_G0V': myInt.I('G','0','V'),
#     'I_GnV': myInt.I('G','n','V'),

#     # edge
#     'I_F0E': myInt.I('F','0','E'),
#     'I_FnE': myInt.I('F','n','E'),
#     'I_G0E': myInt.I('G','0','E'),
#     'I_GnE': myInt.I('G','n','E')
#     }
    

#     D = {
#     # vert
#     'D_F2nV': myInt.D('F', '2n', 'V'),
#     'D_G2nV': myInt.D('G', '2n', 'V'),

#     # edge
#     'D_F2nE': myInt.D('F', '2n', 'E'),
#     'D_G2nE': myInt.D('G', '2n', 'E')
#     }

#     # phi
#     Phi = {
#     'Phi_0V': myInt.phi('0','V'),
#     'Phi_nV': myInt.phi('n','V'),
#     'Phi_0E': myInt.phi('0','E'),
#     'Phi_nE': myInt.phi('n','E')
#     }
#     # psi
#     Psi = {
#     'Psi_0V': myInt.psi('0','V'),
#     'Psi_nV': myInt.psi('n','V'),
#     'Psi_0E': myInt.psi('0','E'),
#     'Psi_nE': myInt.psi('n','E')
#     }

#     # map variables
#     Phi_vars = create_map_variables(myInt, prob)['Phi_vars']
#     Psi_vars = create_map_variables(myInt, prob)['Psi_vars']

#     # other variables
#     z_vars = create_other_decision_variables(myInt)['z']
#     map_product_vars = create_other_decision_variables(myInt)['map_product']

#     # minmax variable
#     minmax_var = set_objective_function(prob)

#     for i in func_vals:
#         for j in ['V', 'E']:
#             for k in ['F', 'G']:
#                 # multiply inclusion matrices
#                 i_n_i_0 = I['I_'+k+'n'+j][i].get_array() @ I['I_'+k+'0'+j][i].get_array()
                

#                 # set maps
#                 if k == 'F':
#                     map_1 = Psi['Psi_n'+j][i]
#                     map_2 = Phi['Phi_0'+j][i]
#                     map_1_vars = Psi_vars[j]['n'][i]
#                     map_2_vars = Phi_vars[j]['0'][i]
#                     map_product_var = map_product_vars[j]['FGF'][i]
#                     z_var = z_vars[j]['FGF'][i]
#                 else:  
#                     map_1 = Phi['Phi_n'+j][i]
#                     map_2 = Psi['Psi_0'+j][i]
#                     map_1_vars = Phi_vars[j]['n'][i]
#                     map_2_vars = Psi_vars[j]['0'][i]
#                     map_product_var = map_product_vars[j]['GFG'][i]
#                     z_var = z_vars[j]['GFG'][i]
             
#                 # write d
#                 dist = D['D_'+k+'2n'+j][i].get_array()
#                 # set the dimensions 
#                 shape_m_tri = dist.shape[0]
#                 shape_n_tri = map_1.get_array().shape[1]
#                 shape_o_tri = map_2.get_array().shape[1]

                
#                 #  constraint 1: loss is bigger than the absolute value of each matrix elements
#                 for a in range(shape_m_tri):
#                     prob += minmax_var >= pulp.lpSum(dist[b, a] * (i_n_i_0[a, d] - map_product_var[a, d]) for b in range(shape_m_tri) for d in range(shape_o_tri))
#                     prob += -minmax_var <= pulp.lpSum(dist[b, a] * (i_n_i_0[a, d] - map_product_var[a, d]) for b in range(shape_m_tri) for d in range(shape_o_tri))

                

#                 # constraints 2: map_multiplication and z relation
#                 for a in range(shape_m_tri):
#                     for d in range(shape_o_tri):
#                         prob += map_product_var[a, d] == pulp.lpSum(z_var[a, c, d] for c in range(shape_n_tri))


#                 # constraint 3: z is less than either of the map values and greater than sum of the map values minus 1
#                 for a in range(shape_m_tri):
#                     for c in range(shape_n_tri):
#                         for d in range(shape_o_tri):
#                             prob += z_var[a, c, d] <= map_1_vars[a, c]
#                             prob += z_var[a, c, d] <= map_2_vars[c, d]
#                             prob += z_var[a, c, d] >= map_1_vars[a, c] + map_2_vars[c, d] - 1

#                 # constraint 4: each column sums to 1
#                 for c in range(shape_n_tri):
#                     prob += pulp.lpSum(map_1_vars[a, c] for a in range(shape_m_tri)) == 1

#                 for d in range(shape_o_tri):
#                     prob += pulp.lpSum(map_2_vars[c, d] for c in range(shape_n_tri)) == 1



# def create_ilp_problem(myInt):
#     """
#     Create the ILP optimization problem.
    
#     Parameters
    
#         myInt: Interleave object
#         The assignment that we want to optimize.
        
#     Returns
    
#         prob: pulp.LpProblem
#         The ILP optimization problem.
#     """
    
#     prob = pulp.LpProblem("Interleave Optimization Problem", pulp.LpMinimize)
    

#     # Set the objective function
#     set_objective_function(prob)
    
#     # Set the constraints
#     set_triangle_constraints(myInt, prob)

#     pulp.LpStatus[prob.status]

#     prob.writeLP("model.lp")  # Write the model in LP format


#     # solve the problem
#     prob.solve()

#     print("status:", pulp.LpStatus[prob.status])
    
    

