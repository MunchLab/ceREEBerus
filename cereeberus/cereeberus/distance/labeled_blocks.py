import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

class LabeledMatrix:
    """
    A class to store a matrix with row and column labels. The matrix is stored as a numpy array, and the row and columns are stored as lists. For example, we could store the 2 x 3 matrix below as a LabeledMatrix.

    +-------+---+---+---+
    |       | a | b | c |
    +=======+===+===+===+
    | **u** | 0 | 1 | 0 |
    +-------+---+---+---+
    | **v** | 1 | 0 | 0 |
    +-------+---+---+---+

    Where the matrix would be initialized by 

    .. code-block:: python

        Mat = LabeledMatrix([[0, 1, 0], [1, 0, 0]], ['u', 'v'], ['a', 'b', 'c'])

    This class supports matrix multiplication, addition, and subtraction, as well as transposition and getting the maximum (absolute) value of the matrix. Note that calling for an entry in the matrix is done by the labels of the row and column, not the indices. For example, to get the entry in the first row and second column from the example above, you could call either of the following options. 

    .. code-block:: python

        Mat['u', 'b']
        Mat.array[0, 1]

    Visualization can be done with the draw method, which will draw the matrix with the row and column labels.

    .. code-block:: python 

        Mat.draw(colorbar = True)

    .. figure:: ../../images/labeled_matrix_ex_Mat.png
        :figwidth: 400px    

    """
    def __init__(self, array = None, rows = None, cols = None, map_dict = None):
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
        elif map_dict is not None:
            self.array = np.zeros((len(rows), len(cols)))
            for c in cols:
                r = map_dict[c]
                # print (f"r = {r}, c = {c}")
                # print(f"rows: {rows}, cols: {cols}")
                self.array[rows.index(r), cols.index(c)] = 1
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
            ValueError: If the matrix sizes are incompatable, or if they are but the row labels of the first matrix do not match the column labels of the second (including ordering).
        """

        if isinstance(other, LabeledBlockMatrix):
            other = other.to_labeled_matrix()
        elif not isinstance(other, LabeledMatrix):
            raise ValueError("Can only multiply with another LabeledMatrix or LabeledBlockMatrix")
        
        if self.cols != other.rows:
            raise ValueError(f"Matrix labels do not match.\nCols of first matrix: {self.cols}\nRows of second matrix: {other.rows}")

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

        if not (self.rows == other.rows and self.cols == other.cols):
            raise ValueError("Cannot add matrices with different row or column labels.")
        
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

    def __getitem__(self, key):
        """
        Get an item from the matrix, where ``key = [row_key, col_key]`` are the **labels** of the row and columns respectively.

        Parameters:
            key: The keys for the row and column to get from the matrix.

        Returns:
            The item from the matrix.
        """
        row_key, col_key = key
        i = self.rows.index(row_key)
        j = self.cols.index(col_key)
        return self.array[i,j]
    
    def __setitem__(self, key , value):
        """
        Set an item in the matrix, where ``key = [row_key, col_key]`` are the **labels** of the row and columns respectively.

        Parameters:
            key: The row and column keys to set in the matrix.
            value: The value to set.
        """
        row_key, col_key = key
        i = self.rows.index(row_key)
        j = self.cols.index(col_key)
        self.array[i,j] = value

    def get_array(self):
        """
        Get the array of the matrix. Works on the big matrix as well as the blocks.

        Returns:
            np.array: The array of the matrix.
        """
        return self.array

    
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

    def min(self):
        """
        Get the minimum value of the matrix.

        Returns:
            float: The minimum value of the matrix.
        """
        try :
            return np.min(self.array)
        except:
            return np.nan

    def shape(self):
        """
        Return the shape of the matrix.

        Returns:
            tuple: The shape of the matrix (rows, columns).
        """
        try:
            return self.array.shape
        except:
            return (0,0)

    def size(self):
        """
        Return the size of the matrix.

        Returns:
            int: The size of the matrix.
        """
        try:
            return self.array.size
        except:
            return 0

    def __repr__(self):
        """
        Return a string representation of the LabeledMatrix.

        Returns:
            str: A string representation of the LabeledMatrix.
        """
        return f"LabeledMatrix(\narray=\n{self.array}, \nrows={self.rows}, \ncols={self.cols})"


    def T(self):
        """
        Transpose the matrix.

        Returns:
            LabeledMatrix: The labeled transposed matrix.
        """
        return LabeledMatrix(self.array.T, self.cols, self.rows)

    def to_labeled_block(self, row_val_to_verts, col_val_to_verts):
        """
        Turns this matrix back into a block dictionary.

        Note this assumes that the row and column labels are already in order from the block dictionary.

        Parameters:
            row_val_to_verts : dict
                A dictionary from function values to a list of row objects.
            col_val_to_verts : dict
                A dictionary from function values to a list of column objects.
        
        Returns:
            LabeledBlockMatrix
                A block matrix with the same data as the labeled matrix.
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

    def col_sum(self):
        """
        Returns the sum of each column in the matrix.
        """
        try:
            return self.array.sum(axis = 0)
        except:
            return []

    
    def is_col_sum_1(self):
        """
        Check if the sum of each column is 1.

        Returns:
            bool: True if the sum of each column is 1, False otherwise.
        """
        try:
            return np.all(self.array.sum(axis = 0) == 1)
        except:
            return False
    
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

        if type(self.rows[0]) == tuple:
            row_labels = [ str(row[0]) + ', ' + str(row[1]) for row in self.rows]
            # print('Warning: Row labels are tuples. Only printing entries 0 and 1. ')
            # This is done because in multigraphs, edges are (u,v,count) and we only want to print u and v

        else:
            row_labels = self.rows

        if type(self.cols[0]) == tuple:
            col_labels = [ str(col[0]) + ', ' + str(col[1]) for col in self.cols]
            # print('Warning: Col labels are tuples. Only printing entries 0 and 1. ')
        else:
            col_labels = self.cols

        ax.set_xticks(range(len(self.cols)), col_labels, rotation = 90)
        ax.set_yticks(range(len(self.rows)), row_labels)

        if colorbar:
            plt.colorbar(im, ax = ax)

        return ax

    def to_indicator(self):
        """
        Convert the matrix to an indicator matrix.

        Returns:
            LabeledMatrix: The indicator matrix.
        """
        return LabeledMatrix((self.array > 0).astype(int), self.rows, self.cols)

########################################################################################

class LabeledBlockMatrix:
    """
    A class to store a block matrix with row and column labels. This is built with the assumption that we have a function value for each object in the labels, and we initialize this with a dictionary for rows and columns that sends function value ``i`` to a list of row or column labels, respectively. 
    
    For example, we could store the block matrix below as a ``LabeledBlockMatrix``. The outside rows give the function values of the vertices, and the inside rows and columns give the labels of the vertices.

    +--------+-------+---+---+---+---+---+
    |                |   1   |   2       |
    +                +---+---+---+---+---+
    |                | a | b | c | d | e |
    +========+=======+===+===+===+===+===+
    | **0**  | **u** | 0 | 0 | 0 | 0 | 0 |
    +--------+-------+---+---+---+---+---+
    | **1**  | **v** | 1 | 1 | 0 | 0 | 0 |
    +        +-------+---+---+---+---+---+
    |        | **w** | 0 | 0 | 0 | 0 | 0 |
    +--------+-------+---+---+---+---+---+
    | **2**  | **x** | 0 | 0 | 1 | 1 | 1 |
    +--------+-------+---+---+---+---+---+
    
    The matrix is stored as a dictionary of LabeledMatrix objects, where the keys are the function values of the vertices. For example, we could store the block matrix below as a ``LabeledBlockMatrix``

    .. code-block:: python

        cols_dict = {1: ['a','b'], 2: ['c','d','e']}
        rows_dict = {0: ['u'], 1: ['v','w'], 2: ['x']}
        map_dict = {'a': 'v', 'b':'v', 'c': 'x', 'd':'x', 'e':'x'}
        lbm = LabeledBlockMatrix(map_dict, rows_dict, cols_dict)

    Note that this can either be drawn with entries filled with either 0's or ``np.nan``. 

    .. code-block:: python

        lbm.draw(colorbar = True)
    
    .. figure:: ../../images/labeled_matrix_ex_BlockMat_0.png
        :figwidth: 400px    

    .. code-block:: python

        lbm.draw(filltype = 'nan', colorbar = True)
    
    .. figure:: ../../images/labeled_matrix_ex_BlockMat_nan.png
        :figwidth: 400px    

    This class supports matrix multiplication, addition, and subtraction, as well as transposition and getting the maximum (absolute) value of the matrix. Note that calling for an item returns the i'th block as a ``LabeledMatrix``. So we could get an entry in the matrix by first calling the block number, and then the row and column labels.

    .. code-block:: python
    
            lbm[1]['v', 'b']


    """
    def __init__(self, map_dict = None, rows_dict = None, cols_dict = None, 
                        random_initialize = False, labled_matrix_dict = None,
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
            labeled_matrix_dict : bool
                Whether to initialize the block matrix from a dictionary of LabeledMatrix objects.
            seed : int
                The seed for the random number generator.
        """

        self.blocks = {}

        if labled_matrix_dict is not None: # Initialize from a dictionary of LabeledMatrix objects
            # make sure all the other dicts are none
            if rows_dict is not None or cols_dict is not None or map_dict is not None or random_initialize:
                raise ValueError("Cannot initialize from a dictionary of LabeledMatrix objects and other dictionaries. random initialization must be False.")
            
            for i in labled_matrix_dict.keys():
                self.blocks[i] = labled_matrix_dict[i]

        if rows_dict is not None:
            all_keys = list(set(rows_dict.keys()) | set(cols_dict.keys()))
            all_keys.sort()



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
                        if cols_ and rows_:
                            for c in cols_:
                                try:
                                    r = map_dict[c]
                                except:
                                    print('Error')
                                    print(f"c: {c}")
                                    print(f"map_dict: {map_dict}")
                                    print(f"cols_: {cols_}")
                                    print(f"rows_: {rows_}")
                                    print(f"blocks[i]: {self.blocks[i]}")
                                    ValueError("Column object {c} not in map_dict")
                                self.blocks[i][r,c]  = 1 
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
            all_keys = list(set(self.blocks.keys()) | set(other.blocks.keys()))
            all_keys.sort()
            
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
    
    def shape(self):
        """
        Return the shape of the block matrix.

        Returns:
            tuple: The shape of the block matrix.
        """
        try:
            return (sum([self.blocks[i].shape()[0] for i in self.blocks.keys()]), sum([self.blocks[i].shape()[1] for i in self.blocks.keys()]))
        except:
            return (0,0)



    def T(self):
        """
        Transpose the block matrix.

        Returns:
            LabeledBlockMatrix: The labeled transposed block matrix.
        """
        result = LabeledBlockMatrix()

        for i in self.blocks.keys():
            result[i] = self.blocks[i].T()
        
        return result
    
    def get_array(self, block_index):
        """
        Get the array of the block matrix at the given block index.

        Parameters:
            block_index (int): The block index to get the array from.

        Returns:
            np.array: The array of the block matrix at the given block index.
        """
        return self.blocks[block_index].array

    def max(self):
        """
        Get the maximum value of the block matrix.

        Returns:
            float: The maximum value of the block matrix.
        """
        try :
            return np.nanmax([self.blocks[i].max() for i in self.blocks.keys()])
        except:
            return np.nan
    
    def absmax(self):
        """
        Get the maximum absolute value of the block matrix.

        Returns:
            float: The maximum absolute value of the block matrix.
        """
        try:
            return np.nanmax([self.blocks[i].absmax() for i in self.blocks.keys()])
        except:
            return np.nan

    
    def min(self):
        """
        Get the minimum value of the block matrix.

        Returns:
            float: The minimum value of the block matrix.
        """
        try :
            return np.nanmin([self.blocks[i].min() for i in self.blocks.keys()])
        except:
            return np.nan

    def get_all_block_indices(self):
        """
        Get all the block indices in order.

        Returns:
            list: The block indices.
        """
        L = list(self.blocks.keys())
        L.sort()
        return L

    def min_block_index(self):
        """
        Get the minimum block index.

        Returns:
            int: The minimum block index.
        """
        return min(self.blocks.keys())
    
    def max_block_index(self):
        """
        Get the maximum block index.

        Returns:
            int: The maximum block index.
        """
        return max(self.blocks.keys())
    
    def get_all_rows(self):
        """
        Get all the row labels of the block matrix in order.

        Returns:
            list: The row labels.
        """
        all_keys =  self.get_all_block_indices()

        rows = [self.blocks[i].rows for i in all_keys]

        # flatten the rows list

        return [label for sublist in rows for label in sublist]

    def get_all_cols(self):
        """
        Get all the col labels of the block matrix in order .

        Returns:
            list: The col labels.
        """
        all_keys =  self.get_all_block_indices()

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
        Create a single matrix from provided arrays on the blocks. Modified from `scipy.linalg.block_diag` to put np.nan in the off-diagonal blocks instead of 0's.

        Parameters:
            *arrs : array-like
                The arrays to put on the diagonal.
    
        Returns:
            np.array: The block diagonal matrix.

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
            LabeledMatrix :
                A labeled matrix with the same data as the block matrix.
        """


        all_keys =  self.get_all_block_indices()
        arrays = [ self.blocks[i].array for i in all_keys]

        if filltype == 'zero':
            BigMatrix = block_diag(*arrays)
        elif filltype == 'nan':
            BigMatrix = self.block_diag_NaN(*arrays)

        rows = self.get_all_rows()
        cols = self.get_all_cols()

        return LabeledMatrix(BigMatrix,rows, cols)
    
    def to_indicator(self):
        """
        Convert the block matrix to an indicator matrix.

        Returns:
            LabeledBlockMatrix: The indicator matrix.
        """
        result = LabeledBlockMatrix()

        for i in self.blocks.keys():
            result[i] = self.blocks[i].to_indicator()
        
        return result

    def to_shifted_blocks(self, shift):
        """
        Shift the blocks in the block matrix.

        Parameters:
            shift (int): The amount to shift the blocks.

        Returns:
            LabeledBlockMatrix: The block matrix with shifted blocks.
        """
        result = LabeledBlockMatrix()

        for i in self.blocks.keys():
            result[i + shift] = self.blocks[i]
        
        return result


    def col_sum(self):
        """
        Returns the sum of each column in the block matrix.

        Returns:
            np.array: The sum of each column in the block matrix.
        """
        try:
            sorted_keys = list(self.blocks.keys())
            sorted_keys.sort()
            return np.concatenate([self.blocks[i].col_sum() for i in sorted_keys])
        except:
            return []
    
    def check_column_sum(self, verbose = False):
        """
        Check that the sum of each column is 1.

        Parameters:
            verbose (bool):
                Prints information on which matrices have columns that do not sum to 1 if True. The default is False.

        Returns:
            bool
                True if the columns sum to 1, False otherwise.
        """
        
        for i in self.blocks.keys():
            if not self.blocks[i].is_col_sum_1():
                if verbose:
                    print(f"Column sum is not 1 for block {i}")
                return False
        
        return True

    def draw(self, ax = None, colorbar = False, filltype = 'zero',  **kwargs):
        """
        Draw the block matrix with row and column labels.

        Parameters:
            ax (matplotlib.axes.Axes): The axes to draw the matrix on. If None, the current axes will be used.
            colorbar (bool): Whether to draw a colorbar.
            filltype (str): Either ``zeros`` or ``nan``. If ``zeros``, the off-diagonal blocks will be filled with zeros prior to drawing. If ``nan``, the off-diagonal blocks will be filled with ``np.nan``, resulting in them showing up white.
            **kwargs: Additional keyword arguments to pass to ``ax.matshow``.
        """

        self.to_labeled_matrix(filltype = filltype).draw(ax = ax, colorbar = colorbar, **kwargs)

        return ax

    def get_rows_dict(self):
        """
        Get the rows dictionary.

        Returns:
            dict: The rows dictionary.
        """
        result = {}
        for i in self.blocks.keys():
            result[i] = self.blocks[i].rows
        
        return result
    
    def get_cols_dict(self):
        """
        Get the columns dictionary.

        Returns:
            dict: The columns dictionary.
        """
        result = {}
        for i in self.blocks.keys():
            result[i] = self.blocks[i].cols
        
        return result
    