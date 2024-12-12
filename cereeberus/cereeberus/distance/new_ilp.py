from cereeberus import ReebGraph, MapperGraph, Interleave
import cereeberus.data.ex_mappergraphs as ex_mg
import matplotlib.pyplot as plt
import numpy as np
import pulp  # for ILP optimization

class MapperOptimizerILP:
    """
    A class to optimize the loss function upper bounding the interleave distance between two Mapper graphs. Uses PuLP for ILP optimization. Uses Interleave class to obtain the matrices required for optimization.

    """

    def __init__(self, interleave_obj, verbose=False):
        """
        Initializes the MapperOptimizerILP class with an instance of the
        Interleave class and sets up the ILP problem.

        Parameters
        interleave_obj : Interleave
            An instance of the Interleave class.
        verbose : bool
            If True, prints the optimization status.

        """
        self.myInt = interleave_obj
        self.verbose = verbose

        # set the function values
        self.func_vals = self.myInt.all_func_vals()

        # initalize the plup problem
        self.prob = pulp.LpProblem("Interleaving Distance", pulp.LpMinimize)


        # initialize the variables
        self.Phi_vars = self.initialize_vars('Phi_vars')
        self.Psi_vars = self.initialize_vars('Psi_vars')
        self.z_vars = self.create_z_vars()
        self.map_product_vars = self.create_map_product_vars()
        self.minmax_var = pulp.LpVariable('minmax_var', lowBound=0, cat='Continuous')
        
        

    def initialize_vars(self, var_name) -> dict:
        """
        Initializes the variables for the ILP optimization.

        Parameters

        var_name : str
            The name of the variable to initialize. Can be 'Phi_vars' or 'Psi_vars'.

        Returns

        dict
            A dictionary of the initialized variables.


        """
        vars_dict = {
            block: {thickening: {obj_type: {} for obj_type in ['V', 'E']}
                    for thickening in ['0', 'n']} for block in self.func_vals
        }

        for block in self.func_vals:
            for thickening in ['0', 'n']:
                for obj_type in ['V', 'E']:
                    # get the shape of the array
                    n_rows, n_cols = self.get_array_shape(var_name, thickening, obj_type, block)

                    # initialize the variables
                    vars_dict[block][thickening][obj_type] = pulp.LpVariable.dicts(
                        f'{var_name}_{thickening}{obj_type}_{block}',
                        ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Binary'
                    )

                    # set the initial values
                    self.set_initial_values(var_name, block, thickening, obj_type, vars_dict)
        return vars_dict
    
    def get_array_shape(self, var_name, thickening, obj_type, block):
        """
        Returns the shape of the array for the given variable name, thickening, object type, and block.
        
        Parameters
            var_name : str
                The name of the variable. Can be 'Phi' or 'Psi'.
            thickening : str
                The thickening level. Can be '0' or 'n'.
            obj_type : str
                The object type. Can be 'V' or 'E'.
            block : int
                The block number.
            
        Returns
            tuple
            A tuple containing the shape of the array."""
        
        if var_name == 'Phi_vars':
            array = (self.myInt.phi(thickening, obj_type))[block].get_array()
        elif var_name == 'Psi_vars':
              self.myInt.psi(thickening, obj_type)[block].get_array()
        else:
            raise ValueError('Invalid var_name')

        return array.shape[0], array.shape[1]

    def set_initial_values(self, var_name, block, thickening, obj_type, vars_dict):
        """
        Sets the initial values of the variables in the ILP optimization problem.

        Parameters
        var_name : str
            The name of the variable. Can be 'Phi' or 'Psi'.
        block : int
            The block number.
        thickening : str
            The thickening level. Can be '0' or 'n'.
        obj_type : str
            The object type. Can be 'V' or 'E'.
        vars_dict : dict
            A dictionary of the initialized variables.

        """

        # get the array
        if var_name == 'Phi_vars':
            array = (self.myInt.phi(thickening, obj_type))[block].get_array()
        elif var_name == 'Psi_vars':
          self.myInt.psi(thickening, obj_type)[block].get_array()
        else:
            raise ValueError('Invalid var_name')

        for a in range(array.shape[0]):
            for b in range(array.shape[1]):
                self.prob += vars_dict[block][thickening][obj_type][(a, b)] == array[a][b]

    def create_z_vars(self):
        """
        Creates the z variables for the ILP optimization. Needed for the triangle diagram optimization.

        Returns
        dict
            A dictionary of the initialized z variables.

        """
        z_vars = { block: {starting_map: {obj_type: {} for obj_type in ['V', 'E']}
                    for starting_map in ['F', 'G']} for block in self.func_vals}
        
        for block in self.func_vals:
            for obj_type in ['V', 'E']:
                for starting_map in ['F', 'G']:
                    if starting_map == 'F':
                        n_rows = self.myInt.psi('n', obj_type)[block].get_array().shape[0]
                        n_rowcol = self.myInt.psi('n', obj_type)[block].get_array().shape[1]
                        n_cols = self.myInt.phi('0', obj_type)[block].get_array().shape[1]
                    else:
                        n_rows = self.myInt.phi('n', obj_type)[block].get_array().shape[0]
                        n_rowcol = self.myInt.phi('n', obj_type)[block].get_array().shape[1]
                        n_cols = self.myInt.psi('0', obj_type)[block].get_array().shape[1]

                    z_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts(
                        f'z_{starting_map}{obj_type}_{block}',
                        ((a, b, c) for a in range(n_rows) for b in range(n_rowcol) for c in range(n_cols)), cat='Binary')


        return z_vars
    
    def create_map_product_vars(self):
            """
            Creates the map product variables for the ILP optimization. needed for the triangle diagram optimization.

            Returns
            dict
                A dictionary of the initialized map product variables.

            """
            map_product_vars ={block: {starting_map: {obj_type: {} for obj_type in ['V', 'E']}
                        for starting_map in ['F', 'G']} for block in self.func_vals}
            
            for block in self.func_vals:
                for obj_type in ['V', 'E']:
                    for starting_map in ['F', 'G']:
                        if starting_map == 'F':
                            n_rows = self.myInt.psi('n', obj_type)[block].get_array().shape[0]
                            n_cols = self.myInt.phi('0', obj_type)[block].get_array().shape[1]
                        else:
                            n_rows = self.myInt.phi('n', obj_type)[block].get_array().shape[0]
                            n_cols = self.myInt.psi('0', obj_type)[block].get_array().shape[1]

                        map_product_vars[block][starting_map][obj_type] = pulp.LpVariable.dicts(
                            f'map_product_{starting_map}{obj_type}_{block}',
                            ((a, b) for a in range(n_rows) for b in range(n_cols)), cat='Integer')
                        
            return map_product_vars
    
    # def _get_dimensions(self, starting_map, obj_type, block, diagram_type):
    #     """
    #     Returns the dimensions of the diagram for the given starting map, object type, block, and diagram type.
        
    #     Parameters
    #         starting_map : str
    #             The starting map. Can be 'F' or 'G'.
    #         obj_type : str
    #             The object type. Can be 'V' or 'E'.
    #         block : int
    #             The block number.
    #         diagram_type : str
    #             The diagram type. Can be 'traiangle', 'parallelogram', or 'mix_parallelogram'.
    #             """
        
    #     if diagram_type == 'triangle':
    #         shape_dict = {}
    #         shape_dict['shape_m_tri'] = self.myInt.D(starting_map, '2n', obj_type)[block].get_array().shape[0]
    #         if starting_map == 'F':
    #             shape_dict['shape_n_tri'] = self.myInt.psi('n', obj_type)[block].get_array().shape[1]
    #             shape_dict['shape_o_tri'] = self.myInt.phi('0', obj_type)[block].get_array().shape[1]
    #         elif starting_map == 'G':
    #             shape_dict['shape_n_tri'] = self.myInt.phi('n', obj_type)[block].get_array().shape[1]
    #             shape_dict['shape_o_tri'] = self.myInt.psi('0', obj_type)[block].get_array().shape[1]
    #         else:
    #             raise ValueError('Invalid starting map')
            
    #         return shape_dict
        
    #     elif diagram_type == 'parallelogram':
    #         shape_dict = {}
    #         if starting_map == 'F':
    #             shape_dict['shape_m_par'] = self.myInt.D('G', '2n', obj_type)[block].get_array().shape[0]
    #             shape_dict['shape_n_par'] = self.myInt.phi('n', obj_type)[block].get_array().shape[1]
    #             shape_dict['shape_o_par'] = self.myInt.I('G', 'n', obj_type)[block].get_array().shape[1]
    #             shape_dict['shape_p_par'] = self.myInt.phi('0', obj_type)[block].get_array().shape[1]
    #         elif starting_map == 'G':
    #             shape_dict['shape_m_par'] = self.myInt.D('F', '2n', obj_type)[block].get_array().shape[0]
    #             shape_dict['shape_n_par'] = self.myInt.psi('n', obj_type)[block].get_array().shape[1]
    #             shape_dict['shape_o_par'] = self.myInt.I('F', 'n', obj_type)[block].get_array().shape[1]
    #             shape_dict['shape_p_par'] = self.myInt.psi('0', obj_type)[block].get_array().shape[1]
    #         else:
    #             raise ValueError('Invalid starting map')
            
    #         return shape_dict
    #     elif diagram_type == 'mix_parallelogram':


    
    def set_objective(self):
        """
        Sets the objective function for the ILP optimization problem.
            
        """
        self.prob += self.minmax_var

    def set_constraints(self):
        """
        Sets the constraints for the ILP optimization problem.

        """
        for block in self.func_vals:
            for starting_map in ['F', 'G']:
                # set the other map based on starting map 
                if starting_map == 'F':
                    other_map = 'G'
                else:
                    other_map = 'F'

                for up_or_down in ['up', 'down']: # deals with 1 (up, down) and 2 (up, down)
                
                #set the matrices
                    if up_or_down == 'up': #NOTE: the change in block indices

                        if block == self.func_vals[-1]: # skip the last block for the up case
                            continue
                        
                        dist_n_other =self.myInt.D(other_map, 'n', 'V')[block+1].get_array()
                        bou_n =self.myInt.B_up(other_map, 'n')[block].get_array()
                        bou_0 =self.myInt.B_up(starting_map, '0')[block].get_array()
                        if starting_map == 'F':
                            map_V =self.myInt.phi('0', 'V')[block+1].get_array()
                            map_E =self.myInt.phi('0', 'E')[block].get_array()
                            map_V_vars =self.Phi_vars[block+1]['0']['V']
                            map_E_vars =self.Phi_vars[block]['0']['E']
                        else:
                            map_V =self.myInt.psi('0', 'V')[block+1].get_array()
                            map_E =self.myInt.psi('0', 'E')[block].get_array()
                            map_V_vars =self.Psi_vars[block+1]['0']['V']
                            map_E_vars =self.Psi_vars[block]['0']['E']
                    else:
                        dist_n_other =self.myInt.D(other_map, 'n', 'V')[block].get_array()
                        bou_n =self.myInt.B_down(other_map, 'n')[block].get_array()
                        bou_0 =self.myInt.B_down(starting_map, '0')[block].get_array()
                        if starting_map == 'F':
                            map_V =self.myInt.phi('0', 'V')[block].get_array()
                            map_E =self.myInt.phi('0', 'E')[block].get_array()
                            map_V_vars =self.Phi_vars[block]['0']['V']
                            map_E_vars =self.Phi_vars[block]['0']['E']
                        else:
                            map_V =self.myInt.psi('0', 'V')[block].get_array()
                            map_E =self.myInt.psi('0', 'E')[block].get_array()
                            map_V_vars =self.Psi_vars[block]['0']['V']
                            map_E_vars =self.Psi_vars[block]['0']['E']

                    # set the dimensions
                    shape_m_mix = dist_n_other.shape[0]
                    shape_n_mix = map_V.shape[1]
                    shape_o_mix = bou_n.shape[1]
                    shape_p_mix = map_E.shape[1]

                    # ƒconstraint 1: loss is bigger than the absolute value of each matrix elements

                    for i in range(shape_m_mix):
                        for k in range(shape_p_mix):
                            # inner difference
                            first_term = pulp.lpSum([map_V_vars[i,j] * bou_0[j][k] for j in range(shape_n_mix)])
                            second_term = pulp.lpSum([bou_n[i][l] * map_E_vars[l,k] for l in range(shape_o_mix)])

                            # total expression
                            expression = pulp.lpSum(dist_n_other[i][h] * (first_term - second_term) for h in range(shape_m_mix))
                            
                            prob += self.minmax_var >= expression
                            prob += -self.minmax_var <= expression

                    # constraint 2: each column sums to 1
                    for j in range(shape_n_mix):
                        prob += pulp.lpSum(map_V_vars[h,j] for h in range(shape_m_mix)) == 1   

                    for k in range(shape_p_mix):
                        prob += pulp.lpSum(map_E_vars[l, k] for l in range(shape_o_mix)) == 1



                for obj_type in ['V', 'E']: # deals with 3, 4, 5, 6, 7, 8, 9, 10
                    # multiply inclusion matrices. Needed for the triangles
                    i_n_i_0 =self.myInt.I(starting_map, 'n', obj_type)[block].get_array() @self.myInt.I(starting_map, '0', obj_type)[block].get_array()

                    # write inclusion matrices for easier reference. Needed for the parallelograms
                    inc_0_para =self.myInt.I(starting_map, '0', obj_type)[block].get_array()
                    inc_n_para =self.myInt.I(other_map, 'n', obj_type)[block].get_array()

                    # write dist matrix for easier reference.
                    # for triangles
                    dist_2n_starting =self.myInt.D(starting_map, '2n', obj_type)[block].get_array()
                    # for parallelograms
                    dist_2n_other =self.myInt.D(other_map, '2n', obj_type)[block].get_array()
                    
                    # set map matrices for easier reference
                    if starting_map == 'F':
                        map_0_para_vars =self.Phi_vars[block]['0'][obj_type]
                        map_n_para_vars =self.Phi_vars[block]['n'][obj_type]

                    if starting_map == 'G':
                        map_0_para_vars =self.Psi_vars[block]['0'][obj_type]
                        map_n_para_vars =self.Psi_vars[block]['n'][obj_type]

                    # set the dimensions
                    shape_m_tri = dist_2n_starting.shape[0] # for triangles
                    shape_m_para = dist_2n_other.shape[0] # for parallelograms

                    shape_o_para =self.myInt.I(other_map, 'n', obj_type)[block].get_array().shape[1] # for parallelograms


                    if starting_map == 'F':
                        shape_n_tri =self.myInt.psi('n', obj_type)[block].get_array().shape[1] # for triangles
                        shape_o_tri =self.myInt.phi('0', obj_type)[block].get_array().shape[1] # for triangles

                        shape_n_para =self.myInt.phi('n', obj_type)[block].get_array().shape[1] # for parallelograms
                        
                        shape_p_para =self.myInt.phi('0', obj_type)[block].get_array().shape[1] # for parallelograms
                    else:
                        shape_n_tri =self.myInt.phi('n', obj_type)[block].get_array().shape[1] # for triangles
                        shape_o_tri =self.myInt.psi('0', obj_type)[block].get_array().shape[1] # for triangles

                        shape_n_para =self.myInt.psi('n', obj_type)[block].get_array().shape[1] # for parallelograms
                        shape_p_para =self.myInt.psi('0', obj_type)[block].get_array().shape[1] # for parallelograms


                    


                    #  constraint 1: loss is bigger than the absolute value of each matrix elements
                
                    # for triangles
                    for  h in range(shape_m_tri):                    
                        prob += self.minmax_var >= pulp.lpSum(dist_2n_starting[i,h] * (i_n_i_0[h,k] - self.map_product_vars[block][starting_map][obj_type][h,k]) for i in range(shape_m_tri) for k in range(shape_o_tri))
                        prob += -self.minmax_var <= pulp.lpSum(dist_2n_starting[i,h] * (i_n_i_0[h,k] - self.map_product_vars[block][starting_map][obj_type][h,k]) for i in range(shape_m_tri) for k in range(shape_o_tri))

                    # for parallelograms
                    for i in range(shape_m_para):
                        for k in range(shape_p_para):
                            # inner difference
                                first_term = pulp.lpSum([map_n_para_vars[i,j] * inc_0_para[j][k] for j in range(shape_n_para)])
                                second_term = pulp.lpSum([inc_n_para[i][l]  * map_0_para_vars[l,k] for l in range(shape_o_para)])


                                # total expression
                                expression = pulp.lpSum(dist_2n_other[i][h] * (first_term - second_term) for h in range(shape_m_para))

                                prob += self.minmax_var >= expression
                                prob += -self.minmax_var <= expression


                    # constraint 2: map_multiplication and z relation. This is for triangles
                    for i in range(shape_m_tri):
                        for k in range(shape_o_tri):
                            prob += self.map_product_vars[block][starting_map][obj_type][i,k] == pulp.lpSum(self.z_vars[block][starting_map][obj_type][i,j,k] for j in range(shape_n_tri))

                    # constraint 3: z is less than either of the map values and greater than sum of the map values minus 1. This is for triangles
                    for i in range(shape_m_tri):
                        for j in range(shape_n_tri):
                            for k in range(shape_o_tri):
                                if starting_map == 'F':
                                    prob += self.z_vars[block][starting_map][obj_type][i,j,k] <=self.Psi_vars[block]['n'][obj_type][i,j]
                                    prob += self.z_vars[block][starting_map][obj_type][i,j,k] <=self.Phi_vars[block]['0'][obj_type][j,k]
                                    prob += self.z_vars[block][starting_map][obj_type][i,j,k] >=self.Psi_vars[block]['n'][obj_type][i,j] +self.Phi_vars[block]['0'][obj_type][j,k] - 1
                                else:
                                    prob += self.z_vars[block][starting_map][obj_type][i,j,k] <=self.Phi_vars[block]['n'][obj_type][i,j]
                                    prob += self.z_vars[block][starting_map][obj_type][i,j,k] <=self.Psi_vars[block]['0'][obj_type][j,k]
                                    prob += self.z_vars[block][starting_map][obj_type][i,j,k] >=self.Phi_vars[block]['n'][obj_type][i,j] +self.Psi_vars[block]['0'][obj_type][j,k] - 1

                    # constraint 4: each column sums to 1
                    if starting_map == 'F':
                        # for triangles
                        for j in range(shape_n_tri):
                            prob += pulp.lpSum(self.Psi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_tri)) == 1                        
                        for k in range(shape_o_tri):
                            prob += pulp.lpSum(self.Phi_vars[block]['0'][obj_type][j,k] for j in range(shape_n_tri)) == 1

                        # for parallelograms
                        for j in range(shape_n_para):
                            prob += pulp.lpSum(self.Phi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_para)) == 1

                        for k in range(shape_p_para):
                            prob += pulp.lpSum(self.Phi_vars[block]['0'][obj_type][j,k] for j in range(shape_o_para)) == 1

                    else:
                        # for triangles
                        for j in range(shape_n_tri):
                            prob += pulp.lpSum(self.Phi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_tri)) == 1

                        for k in range(shape_o_tri):
                            prob += pulp.lpSum(self.Psi_vars[block]['0'][obj_type][j,k] for j in range(shape_n_tri)) == 1

                        # for parallelograms
                        for j in range(shape_n_para):
                            prob += pulp.lpSum(self.Psi_vars[block]['n'][obj_type][i,j] for i in range(shape_m_para)) == 1

                        for k in range(shape_p_para):
                            prob += pulp.lpSum(self.Psi_vars[block]['0'][obj_type][j,k] for j in range(shape_o_para)) == 1

    def solve(self):
        """
        Solves the ILP optimization problem.

        Returns
        float
            The value of the objective function.

        """
        self.set_objective()
        self.prob.solve()
        if self.verbose:
            print(pulp.LpStatus[self.prob.status])

        return pulp.value(self.minmax_var)



                
