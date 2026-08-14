import uuid
import warnings
from typing import Any, Hashable, Iterator, Optional


class Branch:
    """A branch of the decomposition, stored as a single node in a doubly linked list.

    Each branch has a lower and upper function value, which define the branch's
    height interval. A UUID is used as a stable external identifier, while the
    linked-list traversal itself is done through the object pointers.
    """

    def __init__(self, f_low: float, f_high: float, key: Optional[uuid.UUID] = None):
        if f_low == f_high:
            warnings.warn(
                "f_low is equal to f_high; this is a degenerate branch and may indicate an issue with the decomposition.",
                UserWarning,
                stacklevel=2,
            )
        elif f_high < f_low:
            raise ValueError("f_high must be greater than or equal to f_low")

        self.f_low = float(f_low)
        self.f_high = float(f_high)
        self.key = key if key is not None else uuid.uuid4()
        self.prev = None
        self.next = None
        self.top_branch = None
        self.bottom_branch = None

    def __repr__(self) -> str:
        return (
            f"Branch(key={self.key}," 
            f"(f_low, f_high)={self.f_low, self.f_high}, "
            f"bottom_branch = ({self.bottom_branch}), "
            f"top_branch = ({self.top_branch}) )"
        )


class BranchDecomp:
    """
    The branch decomposition of a Reeb graph, stored as a doubly linked list with UUID-based lookup.
    """

    def __init__(self):
        self.head: Optional[Branch] = None
        self.tail: Optional[Branch] = None
        self.by_key: dict[uuid.UUID, Branch] = {}
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Branch]:
        current = self.head
        while current is not None:
            yield current
            current = current.next

    def __getitem__(self, index: int) -> Branch:
        if index < 0:
            index += self._size

        if index < 0 or index >= self._size:
            raise IndexError("list index out of range")

        current = self.head
        for _ in range(index):
            current = current.next
        return current

    def __contains__(self, key: Hashable) -> bool:
        return key in self.by_key

    def _index_of_branch(self, target: Optional[Branch]) -> Optional[int]:
        '''
        Gets the index of a branch in the list. Returns None if the branch is not found.
        '''
        if target is None:
            return None

        for index, branch in enumerate(self):
            if branch.key == target.key:
                return index
        return None

    def __str__(self) -> str:
        lines = []
        for index, branch in enumerate(self):
            low_index = self._index_of_branch(branch.bottom_branch)
            high_index = self._index_of_branch(branch.top_branch)
            lines.append(
                f"Branch {index}: \n\tf_low={branch.f_low}, f_high={branch.f_high}, "
                f"\n\tattach: low={low_index}, high={high_index}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def _resolve_branch_ref(self, ref: Optional[Branch | uuid.UUID | int]) -> Optional[Branch]:
        if ref is None:
            return None
        if isinstance(ref, Branch):
            return ref
        if isinstance(ref, uuid.UUID):
            return self.by_key[ref]
        if isinstance(ref, int):
            if ref < 0 or ref >= self._size:
                raise IndexError("branch index out of range")
            return self[ref]
        raise TypeError("branch reference must be a Branch, UUID, int, or None")

    def append(
        self,
        f_low: float,
        f_high: float,
        top_branch: Optional[Branch | uuid.UUID | int] = None,
        bottom_branch: Optional[Branch | uuid.UUID | int] = None,
        key: Optional[uuid.UUID] = None,
    ) -> Branch:
        """
        Appends a new branch to the end of the list.

        Parameters:
        - f_low: The lower function value of the branch.
        - f_high: The upper function value of the branch.
        - top_branch: Optional reference to the top branch (can be a Branch, UUID, or index).
        - bottom_branch: Optional reference to the bottom branch (can be a Branch, UUID, or index).
        - key: Optional UUID for the branch. If not provided, a new UUID will be generated.

        Returns:
        - The newly created Branch object.

        """

        node = Branch(f_low=f_low, f_high=f_high, key=key)
        node.top_branch = self._resolve_branch_ref(top_branch)
        node.bottom_branch = self._resolve_branch_ref(bottom_branch)
        self.by_key[node.key] = node

        if self.tail is None:
            self.head = node
            self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node

        self._size += 1
        return node

    def insert_before(
        self,
        target: Branch,
        f_low: float,
        f_high: float,
        key: Optional[uuid.UUID] = None,
        top_branch: Optional[Branch | uuid.UUID | int] = None,
        bottom_branch: Optional[Branch | uuid.UUID | int] = None,
    ) -> Branch:
        """
        Inserts a new branch before the specified target branch.

        Parameters:
        - target: The branch before which to insert the new branch.
        - f_low: The lower function value of the new branch.
        - f_high: The upper function value of the new branch.
        - key: Optional UUID for the new branch. If not provided, a new UUID will be generated.
        - top_branch: Optional reference to the top branch (can be a Branch, UUID, or index).
        - bottom_branch: Optional reference to the bottom branch (can be a Branch, UUID, or index).

        Returns:
        - The newly created Branch object.
        """
        if target is self.head:
            node = Branch(f_low=f_low, f_high=f_high, key=key)
            node.top_branch = self._resolve_branch_ref(top_branch)
            node.bottom_branch = self._resolve_branch_ref(bottom_branch)
            self.by_key[node.key] = node
            node.next = target
            target.prev = node
            self.head = node
            self._size += 1
            return node

        node = Branch(f_low=f_low, f_high=f_high, key=key)
        node.top_branch = self._resolve_branch_ref(top_branch)
        node.bottom_branch = self._resolve_branch_ref(bottom_branch)
        self.by_key[node.key] = node
        node.prev = target.prev
        node.next = target
        target.prev.next = node
        target.prev = node
        self._size += 1
        return node

    def insert_after(
        self,
        target: Branch,
        f_low: float,
        f_high: float,
        key: Optional[uuid.UUID] = None,
        top_branch: Optional[Branch | uuid.UUID | int] = None,
        bottom_branch: Optional[Branch | uuid.UUID | int] = None,
    ) -> Branch:
        """
        Inserts a new branch after the specified target branch.

        Parameters:
        - target: The branch after which to insert the new branch.
        - f_low: The lower function value of the new branch.
        - f_high: The upper function value of the new branch.
        - key: Optional UUID for the new branch. If not provided, a new UUID will be generated.
        - top_branch: Optional reference to the top branch (can be a Branch, UUID, or index).
        - bottom_branch: Optional reference to the bottom branch (can be a Branch, UUID, or index).

        Returns:
        - The newly created Branch object.
        """
        if target is self.tail:
            return self.append(
                f_low=f_low,
                f_high=f_high,
                key=key,
                top_branch=top_branch,
                bottom_branch=bottom_branch,
            )
    
        node = Branch(f_low=f_low, f_high=f_high, key=key)
        node.top_branch = self._resolve_branch_ref(top_branch)
        node.bottom_branch = self._resolve_branch_ref(bottom_branch)
        self.by_key[node.key] = node
        node.prev = target
        node.next = target.next
        target.next.prev = node
        target.next = node
        self._size += 1
        return node

    def remove(self, node: Branch) -> None:
        """
        Removes a branch from the list.

        Parameters:
        - node: The branch to remove.
        """
        if node.key not in self.by_key:
            raise ValueError("node is not in this list")

        if node.prev is not None:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next is not None:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        self.by_key.pop(node.key, None)
        node.prev = None
        node.next = None
        self._size -= 1


    def get_by_key(self, key: uuid.UUID) -> Branch:
        return self.by_key[key]

    def clear(self) -> None:
        self.head = None
        self.tail = None
        self.by_key.clear()
        self._size = 0

    def reconstruct(self):
        """Reconstruct a Reeb graph from the linked-list branch structure.

        Reconstruction proceeds branch-by-branch. If an endpoint attaches to a
        prior branch at a height that does not already exist, the corresponding
        owner branch edge is subdivided and the new endpoint is reused.

        Returns:
            ReebGraph: A reconstructed Reeb graph.
        """
        from .reebgraph import ReebGraph

        if self._size == 0:
            return ReebGraph()

        R = ReebGraph()
        branch_paths = {}
        tol = 1e-12
        counter = 0

        def new_vertex_name():
            nonlocal counter
            name = counter
            counter += 1
            return name

        def add_vertex_at_height(f_val):
            v = new_vertex_name()
            R.add_node(v, float(f_val), reset_pos=False)
            return v

        def ensure_vertex_on_branch(branch_obj, f_target):
            f_target = float(f_target)
            path = branch_paths[branch_obj.key]

            for v in path:
                if abs(R.f[v] - f_target) <= tol:
                    return v

            if len(path) == 1:
                raise ValueError("Cannot attach at a new height on a degenerate branch")

            for idx in range(len(path) - 1):
                u, v = path[idx], path[idx + 1]
                f_u, f_v = float(R.f[u]), float(R.f[v])
                lo, hi = min(f_u, f_v), max(f_u, f_v)

                if lo - tol <= f_target <= hi + tol:
                    if abs(f_target - f_u) <= tol:
                        return u
                    if abs(f_target - f_v) <= tol:
                        return v

                    w = new_vertex_name()
                    R.subdivide_edge(u, v, w, f_target)
                    path.insert(idx + 1, w)
                    return w

            raise ValueError("Attachment height is not on owner branch")

        for branch in self:
            low_obj = branch.bottom_branch
            high_obj = branch.top_branch

            low_key = branch.key if low_obj is None else low_obj.key
            high_key = branch.key if high_obj is None else high_obj.key

            low_v = (
                add_vertex_at_height(branch.f_low)
                if low_obj is None
                else ensure_vertex_on_branch(low_obj, branch.f_low)
            )
            high_v = (
                add_vertex_at_height(branch.f_high)
                if high_obj is None
                else ensure_vertex_on_branch(high_obj, branch.f_high)
            )

            if low_v != high_v:
                R.add_edge(low_v, high_v, reset_pos=False)
                branch_paths[branch.key] = [low_v, high_v]
            else:
                branch_paths[branch.key] = [low_v]

        R.set_pos_from_f()
        return R

    def draw(self, ax=None, figsize=(12, 8)):
        """
        Draw the branch decomposition with branches ordered left to right.

        Each branch is drawn as a vertical line at its proper function values.
        Endpoint labels show the index of the attached branch in the decomposition.

        Parameters:
            ax: matplotlib axis object. If None, creates a new figure.
            figsize: tuple of figure size (width, height)

        Returns:
            ax: the matplotlib axis object
        """
        if self._size == 0:
            print("No branches to draw.")
            return None

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)

        n_branches = self._size
        x_positions = [i for i in range(n_branches)]

        purple = '#6B4C7A'
        green = '#5A8C5A'

        for i, branch in enumerate(self):
            x = x_positions[i]
            f_low = branch.f_low
            f_high = branch.f_high

            ax.plot([x, x], [f_low, f_high], 'k-', linewidth=2.5)
            ax.plot(x, f_low, 'o', color=purple, markersize=10)
            ax.plot(x, f_high, 'o', color=green, markersize=10)

            low_ref = branch.bottom_branch
            high_ref = branch.top_branch
            low_index = self._index_of_branch(low_ref)
            high_index = self._index_of_branch(high_ref)

            lower_label = f'(B{low_index})' if low_index is not None else None
            upper_label = f'(B{high_index})' if high_index is not None else None

            if lower_label is not None:
                ax.text(x - 0.12, f_low, lower_label, fontsize=12,
                        ha='right', va='center', color=purple, fontweight='bold')
            if upper_label is not None:
                ax.text(x - 0.12, f_high, upper_label, fontsize=12,
                        ha='right', va='center', color=green, fontweight='bold')

        ax.set_xlabel('Branch', fontsize=14)
        ax.set_ylabel('Function Value', fontsize=14)
        ax.set_title('Branch Decomposition of Reeb Graph', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'B{i}' for i in range(n_branches)], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xlim(-0.5, n_branches - 0.5)

        return ax


