import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

class LabeledMatrix:
    """
    A class to store a matrix with row and column labels.
    """
    def __init__(self, array = None, rows = None, cols = None):
        """
        Initialize the LabeledMatrix with a numpy array, row labels, and column labels.

        Parameters:
            array (array-like): The matrix data.
            rows (list): The labels for the rows.
            cols (list): The labels for the columns.
        """
        self.rows = rows
        self.cols = cols

        if array is not None:
            self.array = np.array(array)
        elif rows is not None and  cols is not None:
            self.array = np.zeros((len(rows), len(cols)))
        else:
            self.array = None

    def __matmul__(self, other):
        """
        Multiply two LabeledMatrix objects using the @ operator.

        Parameters:
            other (LabeledMatrix): The other LabeledMatrix to multiply with.

        Returns:
            LabeledMatrix: The result of the matrix multiplication.

        Raises:
            ValueError: If the other object is not a LabeledMatrix.
        """

        if isinstance(other, LabeledBlockMatrix):
            other = other.to_labeled_matrix()
        elif not isinstance(other, LabeledMatrix):
            raise ValueError("Can only multiply with another LabeledMatrix or LabeledBlockMatrix")
        

        # Perform matrix multiplication
        result_array = self.array @ other.array
        
        # The row labels of the result are the row labels of the first matrix
        result_rows = self.rows
        
        # The column labels of the result are the column labels of the second matrix
        result_cols = other.cols
        
        return LabeledMatrix(result_array, result_rows, result_cols)

    def __add__(self, other):
        """
        Add two LabeledMatrix objects using the + operator.

        Parameters:
            other (LabeledMatrix): The other LabeledMatrix to add.

        Returns:
            LabeledMatrix: The result of the matrix addition.

        Raises:
            ValueError: If the other object is not a LabeledMatrix.
        """
        if not isinstance(other, LabeledMatrix):
            raise ValueError("Can only add with another LabeledMatrix")
        
        # Perform matrix addition
        result_array = self.array + other.array
        
        return LabeledMatrix(result_array, self.rows, self.cols)

    def __sub__(self, other):
        """
        Subtract two LabeledMatrix objects using the - operator.

        Parameters:
            other (LabeledMatrix): The other LabeledMatrix to subtract.

        Returns:
            LabeledMatrix: The result of the matrix subtraction.

        Raises:
            ValueError: If the other object is not a LabeledMatrix.
        """
        if not isinstance(other, LabeledMatrix):
            raise ValueError("Can only subtract with another LabeledMatrix")
        
        # Perform matrix subtraction
        result_array = self.array - other.array
        
        return LabeledMatrix(result_array, self.rows, self.cols)

    def max(self):
        """
        Get the maximum value of the matrix.

        Returns:
            float: The maximum value of the matrix.
        """
        try :
            return np.max(self.array)
        except:
            return np.nan

    def absmax(self):
        """
        Get the maximum absolute value of the matrix.

        Returns:
            float: The maximum absolute value of the matrix.
        """
        try :
            return np.max(np.abs(self.array))
        except:
            return np.nan

    def shape(self):
        """
        Return the shape of the matrix.

        Returns:
            tuple: The shape of the matrix (rows, columns).
        """
        return self.array.shape

    def __repr__(self):
        """
        Return a string representation of the LabeledMatrix.

        Returns:
            str: A string representation of the LabeledMatrix.
        """
        return f"LabeledMatrix(\narray=\n{self.array}, \nrows={self.rows}, \ncols={self.cols})"

    def __getitem__(self, key):
        """
        Get an item from the matrix.

        Parameters:
            key: The key to get from the matrix.

        Returns:
            The item from the matrix.
        """
        return self.array[key]
    
    def __setitem__(self, key, value):
        """
        Set an item in the matrix.

        Parameters:
            key: The key to set in the matrix.
            value: The value to set.
        """
        self.array[key] = value

    def T(self):
        """
        Transpose the matrix.

        Returns:
            LabeledMatrix: The labeled transposed matrix.
        """
        return LabeledMatrix(self.array.T, self.cols, self.rows)

    def f_min (self):
        """
        Return the minimum function value.

        Returns:
            float: The minimum value of the matrix.
        """
        return np.min(list(self.blocks.keys()))
    
    def f_max (self):
        """
        Return the maximum function value.

        Returns:
            float: The maximum value of the matrix.
        """
        return np.max(list(self.blocks.keys()))

    def to_labeled_block(self, row_val_to_verts, col_val_to_verts):
        """
        Turns this matrix back into a block dictionary.

        Note this assumes that the row and column labels are already in order from the block dictionary.

        Parameters:
            row_val_to_verts : dict
                A dictionary from function values to a list of row objects.
            col_val_to_verts : dict
                A dictionary from function values to a list of column objects.
        """

        all_keys = list(row_val_to_verts.keys()) | set(col_val_to_verts.keys())
        all_keys.sort()


        blocks_out = LabeledBlockMatrix()

        curr_row = 0
        curr_col = 0

        for i in all_keys:
            try:
                rows = row_val_to_verts[i]
                next_row = curr_row + len(rows)
            except KeyError:
                rows = []
                next_row = curr_row
            
            try:
                cols = col_val_to_verts[i]
                next_col = curr_col + len(col_val_to_verts[i])
            except KeyError:
                cols = []
                next_col = curr_col 

            A = self.array[curr_row:next_row, curr_col:next_col]
            blocks_out.blocks[i] = LabeledMatrix(A, rows, cols)

            curr_row = next_row
            curr_col = next_col

        return blocks_out
    
    def is_col_sum_1(self):
        """
        Check if the sum of each column is 1.

        Returns:
            bool: True if the sum of each column is 1, False otherwise.
        """
        return np.all(self.array.sum(axis = 0) == 1)
    
    def draw(self, ax = None,  colorbar = False, **kwargs):
        """
        Draw the matrix with row and column labels.

        Parameters:
            ax (matplotlib.axes.Axes): The axes to draw the matrix on. If None, the current axes will be used.
            colorbar (bool): Whether to draw a colorbar.
            **kwargs: Additional keyword arguments to pass to ax.matshow.
        """

        if ax is None:
            ax = plt.gca()
        
        im = ax.matshow(self.array, **kwargs)
            
        # Add vertices as the row and column labels
        ax.set_xticks(range(len(self.cols)), self.cols, rotation = 90)
        ax.set_yticks(range(len(self.rows)), self.rows)

        if colorbar:
            plt.colorbar(im, ax = ax)

        return ax

########################################################################################

class LabeledBlockMatrix:
    """
    A class to store a block matrix with row and column labels.
    """
    def __init__(self, map_dict = None, rows_dict = None, cols_dict = None, 
                        random_initialize = False, 
                        seed = None):
        """
        Initialize the LabeledBlockMatrix with LabeledMatrix objects for each integer.

        rows_dict and cols_dict are dictionaries from integers to lists of row labels and column labels respectively. The keys of the dictionaries are the function values of the vertices. 
        
        If map_dict is provided, it is a dictionary from column objects to row objects. The matrix will be filled in with 1s where the column object maps to the row object. If map_dict is None, the matrix will be filled in with random 1s in each column if random_initialize is True. If none of these are provided, the matrix will be filled with zeros.


        Parameters:
            rows_dict : dict
                A dictionary from integers to lists of row labels.
            cols_dict : dict
                A dictionary from integers to lists of column labels.
            map_dict : dict
                A dictionary from column objects to row objects.
            random_initialize : bool
                Whether to randomly initialize the matrix.
            seed : int
                The seed for the random number generator.
        """

        self.blocks = {}


        if rows_dict is not None:
            all_keys = set(rows_dict.keys()) | set(cols_dict.keys())

            for i in all_keys:
                # i is the function value of the vertices 

                # Initialize the block and fill with empty matrices
                if i in rows_dict:
                    rows_ = rows_dict[i]
                else:
                    rows_ = []

                if i in cols_dict:
                    cols_ = cols_dict[i]
                else:
                    cols_ = []

                
                self.blocks[i] = LabeledMatrix(rows = rows_, cols = cols_)


                # If a map from column objects to row objects is provided, fill in the matrix
                if map_dict is not None:
                    for col_i, label in enumerate(self.blocks[i].cols):
                            row_j = self.blocks[i].rows.index(map_dict[label])
                            self.blocks[i][row_j, col_i] = 1
                elif random_initialize:
                    A = self.blocks[i].array            
                    rng = np.random.default_rng(seed = seed)
                    col_1s = rng.integers(0, A.shape[0], size = A.shape[1])
                    A[col_1s, list(range(A.shape[1]))] = 1
                    self.blocks[i].array = A

    def __repr__(self):
        """
        Return a string representation of the LabeledBlockMatrix.

        Returns:
        str: A string representation of the LabeledBlockMatrix.
        """
        return f"LabeledBlockMatrix(matrices={self.blocks})"

    def __getitem__(self, key):
        """
        Get the i'th block matrix.

        Parameters:
            key: The key to get from the block matrix.

        Returns:
            The item from the block matrix.
        """
        return self.blocks[key]

    def __setitem__(self, key, value):
        """
        Set the i'th block matrix.

        Parameters:
            key: The key to set in the block matrix.
            value: The value to set.
        """
        self.blocks[key] = value
    
    def __matmul__(self, other):
        """
        Multiply two LabeledBlockMatrix objects using the @ operator.

        Parameters:
            other (LabeledBlockMatrix): The other LabeledBlockMatrix to multiply with.

        Returns:
            LabeledBlockMatrix: The result of the matrix multiplication.

        Raises:
            ValueError: If the other object is not a LabeledBlockMatrix.
        """
        if isinstance(other, LabeledBlockMatrix):
        
            # Perform matrix multiplication
            all_keys = set(self.blocks.keys()) | set(other.blocks.keys())
            
            result = LabeledBlockMatrix()

            for i in all_keys:
                if i not in self.blocks:
                    result[i] = LabeledMatrix(cols = other.blocks[i].cols, rows = [])
                elif i not in other.blocks:
                    result[i] = LabeledMatrix(rows = self.blocks[i].rows, cols = [])
                else:
                    result[i] = self.blocks[i] @ other.blocks[i]
            
            return result

        elif isinstance(other, LabeledMatrix):
            return self.to_labeled_matrix() @ other

        else:
            raise ValueError("Can only multiply with another LabeledBlockMatrix or LabeledMatrix")
    
    def __add__(self, other):
        """
        Add two LabeledBlockMatrix objects using the + operator.

        Parameters:
            other (LabeledBlockMatrix): The other LabeledBlockMatrix to add.

        Returns:
            LabeledBlockMatrix: The result of the matrix addition.

        Raises:
            ValueError: If the other object is not a LabeledBlockMatrix.
        """
        if not isinstance(other, LabeledBlockMatrix):
            raise ValueError("Can only add with another LabeledBlockMatrix")
        
        # Perform matrix addition
        all_keys = set(self.blocks.keys()) | set(other.blocks.keys())
        
        result = LabeledBlockMatrix()

        for i in all_keys:
            if i not in self.blocks:
                result[i] = other.blocks[i]
            elif i not in other.blocks:
                result[i] = self.blocks[i]
            else:
                result[i] = self.blocks[i] + other.blocks[i]
        
        return result

    def __sub__(self, other):
        """
        Subtract two LabeledBlockMatrix objects using the - operator.

        Parameters:
            other (LabeledBlockMatrix): The other LabeledBlockMatrix to subtract.

        Returns:
            LabeledBlockMatrix: The result of the matrix subtraction.

        Raises:
            ValueError: If the other object is not a LabeledBlockMatrix.
        """
        if not isinstance(other, LabeledBlockMatrix):
            raise ValueError("Can only subtract with another LabeledBlockMatrix")
        
        # Perform matrix subtraction
        all_keys = set(self.blocks.keys()) | set(other.blocks.keys())
        
        result = LabeledBlockMatrix()

        for i in all_keys:
            if i not in self.blocks:
                result[i] = other.blocks[i]
            elif i not in other.blocks:
                result[i] = self.blocks[i]
            else:
                result[i] = self.blocks[i] - other.blocks[i]
        
        return result

    def max(self):
        """
        Get the maximum value of the block matrix.

        Returns:
            float: The maximum value of the block matrix.
        """
        return max([self.blocks[i].max() for i in self.blocks.keys()])
    
    def absmax(self):
        """
        Get the maximum absolute value of the block matrix.

        Returns:
            float: The maximum absolute value of the block matrix.
        """
        return max([self.blocks[i].absmax() for i in self.blocks.keys()])
    
    def get_all_rows(self):
        """
        Get all the row labels of the block matrix in order.

        Returns:
            list: The row labels.
        """
        all_keys = list(self.blocks.keys())
        all_keys.sort()

        rows = [self.blocks[i].rows for i in all_keys]

        # flatten the rows list

        return [label for sublist in rows for label in sublist]

    def get_all_cols(self):
        """
        Get all the col labels of the block matrix in order .

        Returns:
            list: The col labels.
        """
        all_keys = list(self.blocks.keys())
        all_keys.sort()

        cols = [self.blocks[i].cols for i in all_keys]

        return [label for sublist in cols for label in sublist]
    
    def get_all_labels(self):
        """
        Get all the row and column of the block matrix in function value order.

        Returns:
            tuple: The row and column labels.
        """
        return self.get_all_rows(),  self.get_all_cols()

    def block_diag_NaN(self, *arrs):
        """
        Create a block diagonal matrix from provided arrays. Modified from `scipy.linalg.block_diag` to put np.nan in the off-diagonal blocks.


        """
        if arrs == ():
            arrs = ([],)
        arrs = [np.atleast_2d(a) for a in arrs]

        bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
        if bad_args:
            raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args)

        shapes = np.array([a.shape for a in arrs])
        # out_dtype = np.result_type(*[arr.dtype for arr in arrs])
        # out = np.empty(np.sum(shapes, axis=0), dtype=out_dtype)
        out = np.empty(np.sum(shapes, axis=0)) 
        out.fill(np.nan)

        r, c = 0, 0
        for i, (rr, cc) in enumerate(shapes):
            out[r:r + rr, c:c + cc] = arrs[i]
            r += rr
            c += cc
        return out

    def to_labeled_matrix(self, filltype = 'zero'):
        """
        Convert to a single block diagonal matrix.
        
        returns:
            LabeledMatrix 
                A labeled matrix with the same data as the block matrix.
        """

        arrays = [ self.blocks[i].array for i in self.blocks.keys()]

        if filltype == 'zero':
            BigMatrix = block_diag(*arrays)
        elif filltype == 'nan':
            BigMatrix = self.block_diag_NaN(*arrays)
            print(type(BigMatrix))

        rows = self.get_all_rows()
        cols = self.get_all_cols()

        return LabeledMatrix(BigMatrix,rows, cols)


    
    def check_column_sum(self, matrix_dict, verbose = False):
        """
        Check that the sum of each column is 1. TODO NOT YET UPDATED

        Parameters:
            matrix_dict : dict
                Either a dictionary with keys 'rows', 'cols', and 'array' that are lists of rows, columns, and a numpy array respectively), or a block dictionary where keys are function values and output is a dictionary of the above form. 
            verbose : bool
                Prints information on which matrices have columns that do not sum to 1 if True. The default is False.

        Returns:
            bool
                True if the columns sum to 1, False otherwise.
        """
        
        if 'array' in matrix_dict:
            # This will be false if any of the columns does not sum to 1
            check = np.all(matrix_dict['array'].sum(axis = 0) == 1)

            if not check and verbose : 
                print('The columns of the distance matrix do not sum to 1')

        else:
            for i in matrix_dict.keys():
                D_small = matrix_dict[i]
                check = np.all(D_small['array'].sum(axis = 0) == 1) 

                if not check and verbose: 
                    print(f'The columns of the distance matrix for function value {i} do not sum to 1')

        return check

    def draw(self, ax = None, colorbar = False, filltype = 'zero',  **kwargs):
        """
        Draw the block matrix with row and column labels.

        Parameters:
            ax (matplotlib.axes.Axes): The axes to draw the matrix on. If None, the current axes will be used.
            colorbar (bool): Whether to draw a colorbar.
            filltype (str): Either 'zeros' or 'nan'. If 'zeros', the off-diagonal blocks will be filled with zeros. If 'nan', the off-diagonal blocks will be filled with np.nan.
            **kwargs: Additional keyword arguments to pass to ax.matshow.
        """

        self.to_labeled_matrix(filltype = filltype).draw(ax = ax, colorbar = colorbar, **kwargs)

        return ax
