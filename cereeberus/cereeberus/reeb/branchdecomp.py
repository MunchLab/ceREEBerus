import uuid
import warnings
from typing import Any, Hashable, Iterator, Optional

import numpy as np


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
        self._top_branch = None
        self._bottom_branch = None
        # Backward pointers: branches whose top_branch/bottom_branch is this
        # branch, kept in sync automatically by the top_branch/bottom_branch
        # setters below. This lets code that needs to know "who attaches to
        # me" look it up directly instead of scanning the whole decomposition.
        self.attached_via_top: list["Branch"] = []
        self.attached_via_bottom: list["Branch"] = []

    @property
    def top_branch(self) -> Optional["Branch"]:
        return self._top_branch

    @top_branch.setter
    def top_branch(self, new_branch: Optional["Branch"]) -> None:
        old_branch = self._top_branch
        if old_branch is new_branch:
            return
        if old_branch is not None:
            try:
                old_branch.attached_via_top.remove(self)
            except ValueError:
                pass
        if new_branch is not None:
            new_branch.attached_via_top.append(self)
        self._top_branch = new_branch

    @property
    def bottom_branch(self) -> Optional["Branch"]:
        return self._bottom_branch

    @bottom_branch.setter
    def bottom_branch(self, new_branch: Optional["Branch"]) -> None:
        old_branch = self._bottom_branch
        if old_branch is new_branch:
            return
        if old_branch is not None:
            try:
                old_branch.attached_via_bottom.remove(self)
            except ValueError:
                pass
        if new_branch is not None:
            new_branch.attached_via_bottom.append(self)
        self._bottom_branch = new_branch

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

    def __init__(self, reebgraph=None):
        '''
        Initialize an empty branch decomposition. If a Reeb graph is provided, the decomposition is computed immediately.
        '''
        self.head: Optional[Branch] = None
        self.tail: Optional[Branch] = None
        self.by_key: dict[uuid.UUID, Branch] = {}
        self._size = 0
        self.paths: list[list[Any]] = []

        if reebgraph is not None:
            self.decompose(reebgraph)

    @property
    def branches(self) -> np.ndarray:
        """Compatibility view of the decomposition as a (n, 4) array.

        Each row stores the branch interval and the list-order indices of the
        branches attached at the lower and upper endpoints, respectively.
        """
        rows = []
        for branch in self:
            low_idx = self._index_of_branch(branch.bottom_branch) if branch.bottom_branch is not None else self._index_of_branch(branch)
            high_idx = self._index_of_branch(branch.top_branch) if branch.top_branch is not None else self._index_of_branch(branch)
            rows.append((branch.f_low, branch.f_high, low_idx, high_idx))

        if not rows:
            return np.empty((0, 4), dtype=float)
        return np.asarray(rows, dtype=float)

    def get_branch(self, branch_id: int) -> Branch:
        if branch_id < 0 or branch_id >= self._size:
            raise IndexError("branch_id out of range")
        return self[branch_id]

    def get_branch_path(self, branch_id: int) -> list[Any]:
        if branch_id < 0 or branch_id >= len(self.paths):
            raise IndexError("branch_id out of range")
        return self.paths[branch_id]

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

    def _validate_attachment_values(
        self,
        f_low: float,
        f_high: float,
        top_branch: Optional[Branch],
        bottom_branch: Optional[Branch],
    ) -> None:
        """Ensure attached endpoints lie strictly inside their owner intervals."""
        if top_branch is not None and not (
            top_branch.f_low < f_high < top_branch.f_high
        ):
            raise ValueError(
                "Top attachment requires top_branch.f_low < f_high < "
                "top_branch.f_high"
            )

        if bottom_branch is not None and not (
            bottom_branch.f_low < f_low < bottom_branch.f_high
        ):
            raise ValueError(
                "Bottom attachment requires bottom_branch.f_low < f_low < "
                "bottom_branch.f_high"
            )

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

        resolved_top = self._resolve_branch_ref(top_branch)
        resolved_bottom = self._resolve_branch_ref(bottom_branch)
        self._validate_attachment_values(
            f_low, f_high, resolved_top, resolved_bottom
        )

        node = Branch(f_low=f_low, f_high=f_high, key=key)
        node.top_branch = resolved_top
        node.bottom_branch = resolved_bottom
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
            resolved_top = self._resolve_branch_ref(top_branch)
            resolved_bottom = self._resolve_branch_ref(bottom_branch)
            self._validate_attachment_values(
                f_low, f_high, resolved_top, resolved_bottom
            )

            node = Branch(f_low=f_low, f_high=f_high, key=key)
            node.top_branch = resolved_top
            node.bottom_branch = resolved_bottom
            self.by_key[node.key] = node
            node.next = target
            target.prev = node
            self.head = node
            self._size += 1
            return node

        resolved_top = self._resolve_branch_ref(top_branch)
        resolved_bottom = self._resolve_branch_ref(bottom_branch)
        self._validate_attachment_values(
            f_low, f_high, resolved_top, resolved_bottom
        )

        node = Branch(f_low=f_low, f_high=f_high, key=key)
        node.top_branch = resolved_top
        node.bottom_branch = resolved_bottom
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
    
        resolved_top = self._resolve_branch_ref(top_branch)
        resolved_bottom = self._resolve_branch_ref(bottom_branch)
        self._validate_attachment_values(
            f_low, f_high, resolved_top, resolved_bottom
        )

        node = Branch(f_low=f_low, f_high=f_high, key=key)
        node.top_branch = resolved_top
        node.bottom_branch = resolved_bottom
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
        # Clear outgoing pointers via the setters so this removed node no
        # longer lingers in another branch's attached_via_top/bottom lists.
        node.top_branch = None
        node.bottom_branch = None
        self._size -= 1


    def get_by_key(self, key: uuid.UUID) -> Branch:
        return self.by_key[key]

    def clear(self) -> None:
        self.head = None
        self.tail = None
        self.by_key.clear()
        self._size = 0
        self.paths.clear()

    @staticmethod
    def _lowest_available_vertex(graph):
        """Return the lowest vertex with outgoing edges, or None if none remain."""
        available_vertices = [v for v in graph.nodes if graph.up_degree(v) > 0]
        if not available_vertices:
            return None
        return min(available_vertices, key=lambda v: graph.f[v])

    @staticmethod
    def _largest_upward_path(graph, start_vertex):
        """Greedily follow upward edges until reaching a local maximum."""
        path = [start_vertex]
        current = start_vertex

        while graph.up_degree(current) > 0:
            next_vertex = next(graph.successors(current))
            path.append(next_vertex)
            current = next_vertex

        return path

    @staticmethod
    def _remove_path_edges(graph, path):
        """Remove the edges in a path from the working graph."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)

    def decompose(self, reebgraph):
        """Decompose a Reeb graph into a linked-list of branches.

        Repeatedly pick the lowest vertex with outgoing edges, take the greedy upward path from that point, and convert it into a branch. Any remaining isolated vertices are stored as degenerate branches with equal endpoint values.
        """
        working = reebgraph.copy()
        self.clear()
        self.paths = []
        endpoint_owner = {}

        while working.number_of_edges() > 0:
            start = self._lowest_available_vertex(working)
            if start is None:
                break

            path = self._largest_upward_path(working, start)
            start_v = path[0]
            end_v = path[-1]

            low_ref = endpoint_owner.get(start_v)
            high_ref = endpoint_owner.get(end_v)

            new_branch = self.append(
                f_low=working.f[start_v],
                f_high=working.f[end_v],
                bottom_branch=low_ref,
                top_branch=high_ref,
            )
            self.paths.append(list(path))

            for v in path:
                endpoint_owner.setdefault(v, new_branch)

            self._remove_path_edges(working, path)

        for v in list(working.nodes):
            if v not in endpoint_owner:
                self.append(f_low=working.f[v], f_high=working.f[v], bottom_branch=None, top_branch=None)
                self.paths.append([v])
                endpoint_owner[v] = self.tail

        return self

    def _resolve_path(self, path) -> list[Branch]:
        """Resolve a path's branch references to branches in this decomposition."""
        if not path:
            raise ValueError("path must contain at least one branch")

        resolved_path = [self._resolve_branch_ref(branch) for branch in path]
        if any(branch is None for branch in resolved_path):
            raise ValueError("path entries cannot be None")
        return resolved_path

    @staticmethod
    def _attachment_value(first: Branch, second: Branch) -> Optional[float]:
        """Return the height of an upward transition, or None when it is invalid."""
        if first.top_branch is second:
            return first.f_high
        if second.bottom_branch is first:
            return second.f_low
        return None

    def check_branch_path(self, path) -> bool:
        """Return whether ``path`` represents a valid upward path through branches.

        Consecutive entries must be connected by either the first branch's top
        attachment or the second branch's bottom attachment. Attachment heights
        must be non-decreasing along the path: equal consecutive heights are
        allowed, since both attachments are then guaranteed (by the strict-interior
        invariant enforced at construction) to land on the same shared point of
        the branch between them -- i.e. several branches meeting at one vertex.
        """
        branches = self._resolve_path(path)
        previous_value = branches[0].f_low

        for first, second in zip(branches, branches[1:]):
            attachment_value = self._attachment_value(first, second)
            if attachment_value is None or attachment_value < previous_value:
                return False
            previous_value = attachment_value

        return True

    def get_func_vals_for_path(self, path) -> list[float]:
        """Return the ordered attachment heights for a valid branch path.

        Path entries may be branch objects, UUIDs, or list-order indices. The
        returned list has one value for each transition in ``path``.
        """
        branches = self._resolve_path(path)
        if not self.check_branch_path(branches):
            raise ValueError("path is not a valid upward branch path")

        return [
            self._attachment_value(first, second)
            for first, second in zip(branches, branches[1:])
        ]

    def find_subpath(self, path, a: float, b: Optional[float] = None) -> list[Branch]:
        """Return branches in a valid upward path that meet the interval ``[a, b]``.

        When ``b`` is omitted, return the one branch containing height ``a``.
        Path endpoint ownership follows the transition convention: a transition
        at the first branch's top belongs to that branch; one at the next
        branch's bottom belongs to the next branch.
        """
        branches = self._resolve_path(path)
        if not self.check_branch_path(branches):
            raise ValueError(f"path is not a valid upward branch path")

        if b is None:
            b = a
        if b < a:
            raise ValueError("b must be greater than or equal to a")

        if b < branches[0].f_low:
            return []

        result = []
        started = False

        for first, second in zip(branches, branches[1:]):
            attachment_value = self._attachment_value(first, second)
            owned_by_first = first.top_branch is second

            if not started:
                if owned_by_first:
                    if a <= attachment_value:
                        result.append(first)
                        started = True
                    else:
                        continue
                elif a < attachment_value:
                    result.append(first)
                    started = True
                else:
                    continue
            else:
                # Once we've started including branches, every subsequent
                # 'first' is the previous pair's 'second' -- a connector we've
                # already committed to passing through, so it must be
                # included regardless of whether it owns any of [a, b] itself
                # (e.g. a zero-width pass-through branch at a shared vertex).
                result.append(first)

            if owned_by_first:
                if b <= attachment_value:
                    return result
            else:
                # This attachment height is owned by 'second' (its bottom), not
                # 'first' -- so if b lands exactly on it, don't return yet;
                # 'second' still needs to be included (via the next iteration
                # or the last_branch fallback below).
                if b < attachment_value:
                    return result

        last_branch = branches[-1]
        if a <= last_branch.f_high:
            result.append(last_branch)

        return result

    def path_image(
        self,
        path,
        a: float,
        b: float,
        branch_map: "BranchDecompMap",
    ) -> list[Branch]:
        """Return the target image of ``path`` restricted to ``[a, b]``.

        ``branch_map`` must map this decomposition to its target decomposition.
        The returned target path is resolved from the map's internally stored
        UUIDs and has adjacent duplicate branches removed.
        """
        if not isinstance(branch_map, BranchDecompMap):
            raise TypeError("branch_map must be a BranchDecompMap")
        if branch_map.source is not self:
            raise ValueError("branch_map source must be this decomposition")
        if b < a:
            raise ValueError("b must be greater than or equal to a")

        source_path = self._resolve_path(path)
        if not self.check_branch_path(source_path):
            raise ValueError("path is not a valid upward branch path")

        source_subpath = self.find_subpath(source_path, a, b)
        if not source_subpath:
            return []

        attachment_values = self.get_func_vals_for_path(source_subpath)
        image_path = []

        for index, source_branch in enumerate(source_subpath):
            low = a if index == 0 else attachment_values[index - 1]
            high = b if index == len(source_subpath) - 1 else attachment_values[index]
            target_path = branch_map.get_image(source_branch)

            if target_path:
                image_path.extend(branch_map.target.find_subpath(target_path, low, high))

        return [
            branch
            for index, branch in enumerate(image_path)
            if index == 0 or branch is not image_path[index - 1]
        ]

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

            if branch.f_low == branch.f_high:
                if low_obj is not None or high_obj is not None:
                    raise ValueError("Degenerate branches cannot have attachments")
                branch_paths[branch.key] = [add_vertex_at_height(branch.f_low)]
                continue

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

    def smooth(self, eps: float) -> tuple["BranchDecomp", "BranchDecompMap"]:
        """Smooth this branch decomposition by expanding intervals by epsilon.
        
        Given this branch decomposition of a Reeb graph, returns the branch decomposition 
        of the smoothed Reeb graph with parameter epsilon, along with a BranchDecompMap 
        tracking how branches map to paths in the smoothed decomposition.
        
        Args:
            eps (float): The smoothing parameter (must be positive).
        
        Returns:
            tuple: (smoothed_decomp, branch_map) where:
                - smoothed_decomp is a BranchDecomp of the smoothed graph
                - branch_map is a BranchDecompMap tracking the image of each branch
        
        Raises:
            ValueError: If eps <= 0
        
        Notes:
            The (upfork, downfork) case is not fully implemented yet.
        """
        if eps <= 0:
            raise ValueError("Epsilon must be positive.")
        
        B = self
        B_smooth = BranchDecomp()
        eta = BranchDecompMap(B, B_smooth)

        def _find_connecting_path(start, end):
            """Find the (ascending) sequence of branches connecting ``start`` to ``end`` in B_smooth.

            Used when two branches (e.g. two halves of a just-split branch)
            don't attach to each other directly, but still need to be joined
            into a single ascending path -- since B_smooth is built up from a
            connected Reeb graph, some route between any two of its branches
            must already exist via other, unrelated attachments. This walks
            the live graph (top_branch/bottom_branch plus the attached_via_top/
            attached_via_bottom backward pointers) to find it, then returns it
            in ascending-height order (``start`` and ``end`` are not attempted
            to be reordered -- ``start`` must already be the lower one).
            """
            if start is end:
                return [start]

            visited = {start.key: None}
            queue = [start]
            while queue:
                current = queue.pop(0)
                neighbors = [
                    current.top_branch, current.bottom_branch,
                    *current.attached_via_top, *current.attached_via_bottom,
                ]
                for neighbor in neighbors:
                    if neighbor is None or neighbor.key in visited:
                        continue
                    visited[neighbor.key] = current
                    if neighbor is end:
                        path = [end]
                        while path[-1] is not start:
                            path.append(visited[path[-1].key])
                        return list(reversed(path))
                    queue.append(neighbor)

            raise ValueError(
                "no connecting path found between two branches expected to be "
                "in the same connected component of B_smooth"
            )

        def _merge_overlapping(first, second):
            """Concatenate two ascending paths that may re-derive the same shared tail.

            Used when a vanishing branch's up-path and down-path images both
            resolve through the same underlying structure (since both sides
            converge once the branch collapses) -- naively concatenating them
            would repeat that shared portion. If ``second`` re-enters at some
            branch already in ``first``, drop everything up through that
            branch from ``second`` before joining.
            """
            if not first or not second:
                return list(first) + list(second)
            for idx, branch in enumerate(second):
                if branch is first[-1]:
                    return list(first) + list(second[idx + 1:])
            return list(first) + list(second)

        def _attach_branch_at_height(path, height):
            """Find the branch in an image path whose interior strictly contains ``height``.

            Invariant: an attachment point must always be strictly interior to
            the branch it attaches into (top_branch.f_low < f_high < top_branch.f_high,
            and symmetrically for bottom_branch) -- omitting an attachment
            (None) is only valid at a genuine local min/max. This is enforced
            at construction time by _validate_attachment_values, which lets us
            resolve ties deterministically instead of guessing: whenever two
            branches in the path meet at a transition height, exactly one of
            them is guaranteed to have that height strictly interior.
              - If first.top_branch is second, then second.f_low < first.f_high
                < second.f_high was required when `first` was created, so the
                height is guaranteed interior to `second`, never to `first`.
              - If second.bottom_branch is first, then first.f_low < second.f_low
                < first.f_high was required when `second` was created, so the
                height is guaranteed interior to `first`, never to `second`.
            """
            if not path:
                return None

            branch = path[-1]
            for first, second in zip(path, path[1:]):
                attachment_value = BranchDecomp._attachment_value(first, second)
                if height < attachment_value:
                    branch = first
                    break
                if height == attachment_value:
                    branch = second if first.top_branch is second else first
                    break

            if not (branch.f_low < height < branch.f_high):
                raise ValueError(
                    f"height {height} is not strictly interior to any branch in "
                    "the path; this likely indicates a degenerate height that "
                    "coincides exactly with a local min/max, which is not yet supported."
                )
            return branch

        def _bridge_append(image_slice, attach_branch):
            """Append attach_branch if the slice's owner-convention boundary stopped one hop short of it.

            find_subpath's slicing convention can end a slice at a branch that
            only touches the requested height at its own boundary, one hop
            before the branch where that height is actually interior (the one
            _attach_branch_at_height returns). When that happens, the two are
            still directly adjacent, so bridge the gap by including it.
            """
            if attach_branch is None or not image_slice or image_slice[-1] is attach_branch:
                return image_slice
            last = image_slice[-1]
            if last.top_branch is attach_branch or attach_branch.bottom_branch is last:
                image_slice.append(attach_branch)
            return image_slice

        def _bridge_prepend(image_slice, attach_branch):
            """Prepend attach_branch if the slice's owner-convention boundary starts one hop past it."""
            if attach_branch is None or not image_slice or image_slice[0] is attach_branch:
                return image_slice
            first = image_slice[0]
            if attach_branch.top_branch is first or first.bottom_branch is attach_branch:
                image_slice.insert(0, attach_branch)
            return image_slice

        def _repoint_stale_eta_paths(old_branch, low_half, high_half, split_height, bridge):
            """Rewrite already-stored eta image paths that still reference a removed, split branch.

            ``old_branch`` (e.g. B_later) has been removed from B_smooth and
            replaced by ``low_half``/``high_half``. Any eta[j] path set for a
            previously-processed branch j may still contain old_branch's key
            (eta stores paths as UUIDs, independent of the live linked list),
            which would raise a KeyError once resolved. For each occurrence,
            the neighboring path entries (still valid) tell us the height at
            which the path entered/exited old_branch, which determines whether
            that occurrence belongs to low_half, high_half, or both (if the
            occurrence spans across split_height). ``low_half`` and
            ``high_half`` are not directly attached to each other even after
            the post-split slide -- ``bridge`` is the already-resolved list of
            branches (in ascending height order) that connects them (see
            caller), so a spanning occurrence is replaced by
            ``[low_half] + bridge + [high_half]``.
            """
            old_key = old_branch.key
            target = eta.target
            bridge_keys = [b.key for b in bridge]
            for source_key, path_keys in eta.image_paths.items():
                if old_key not in path_keys:
                    continue

                new_path_keys = []
                n = len(path_keys)
                for idx, key in enumerate(path_keys):
                    if key != old_key:
                        new_path_keys.append(key)
                        continue

                    if idx > 0:
                        prev_branch = target.get_by_key(path_keys[idx - 1])
                        entry_height = (
                            old_branch.f_low if old_branch.bottom_branch is prev_branch
                            else prev_branch.f_high
                        )
                    else:
                        entry_height = old_branch.f_low

                    if idx + 1 < n:
                        next_branch = target.get_by_key(path_keys[idx + 1])
                        exit_height = (
                            old_branch.f_high if old_branch.top_branch is next_branch
                            else next_branch.f_low
                        )
                    else:
                        exit_height = old_branch.f_high

                    entry_half = low_half if entry_height <= split_height else high_half
                    exit_half = low_half if exit_height <= split_height else high_half

                    if entry_half is exit_half:
                        new_path_keys.append(entry_half.key)
                    else:
                        new_path_keys.append(low_half.key)
                        new_path_keys.extend(bridge_keys)
                        new_path_keys.append(high_half.key)

                eta.image_paths[source_key] = new_path_keys
        
        for i, b in enumerate(self):
            # print(f"Working on branch {i}: {b.f_low, b.f_high}")
            low = b.f_low
            high = b.f_high
            
            # Case 1: Local min at bottom
            if b.bottom_branch is None:
                # print("Case 1a: Local min/max")
                if b.top_branch is None:
                    new_branch = B_smooth.append(low - eps, high + eps, top_branch=None, bottom_branch=None)
                    eta.set_image(i, [new_branch])
                
                # Case 1b: Local min/down fork
                else:
                    # print("Case 1b: Local min/down fork")
                    # Follow down from top attachment until below high-eps
                    check_attach_top = b.top_branch
                    path_in_old = [check_attach_top]
                    
                    while (check_attach_top.bottom_branch is not None and 
                           check_attach_top.f_low >= high - eps):
                        check_attach_top = check_attach_top.bottom_branch
                        path_in_old.append(check_attach_top)
                    
                    # Reverse path since we followed down but need to go up
                    path_in_old = list(reversed(path_in_old))
                    
                    # The new branch (with local min at bottom) attaches at top to whichever branch
                    # in the image path actually contains the attachment height (high - eps).
                    first_branch_image = eta.get_image(B._index_of_branch(path_in_old[0]))
                    top_attach = _attach_branch_at_height(first_branch_image, high - eps)
                    
                    # Create the new branch with attachment at top
                    new_branch = B_smooth.append(low - eps, high - eps, top_branch=top_attach, bottom_branch=None)
                    
                    # Get image of path in new decomposition
                    image = B.path_image(path_in_old, max(high - eps, low), high, eta)
                    image = _bridge_prepend(image, top_attach)
                    
                    # Add new branch at beginning if needed
                    if high - eps > low:
                        image.insert(0, new_branch)
                    
                    eta.set_image(i, image)
                    
                    # Validate
                    if not B_smooth.check_branch_path(image):
                        warnings.warn(
                            f"local min/downfork case created invalid path for branch {i}: {image}",
                            UserWarning,
                            stacklevel=2
                        )
            
            # Case 2: Up fork at bottom
            else:
                # print("Case 2a: Up fork / local max")
                # Case 2a: Up fork/local max
                if b.top_branch is None:
                    # Follow up from bottom attachment until above low+eps
                    check_attach_bottom = b.bottom_branch
                    path_in_old = [check_attach_bottom]
                    
                    while (check_attach_bottom.top_branch is not None and 
                           check_attach_bottom.f_high < low + eps):
                        check_attach_bottom = check_attach_bottom.top_branch
                        path_in_old.append(check_attach_bottom)
                    
                    # The new branch (with local max at top) attaches at bottom to whichever branch
                    # in the image path actually contains the attachment height (low + eps).
                    last_branch_image = eta.get_image(B._index_of_branch(path_in_old[-1]))
                    bottom_attach = _attach_branch_at_height(last_branch_image, low + eps)
                    
                    # Create the new branch with attachment at bottom (and local max at top)
                    new_branch = B_smooth.append(low + eps, high + eps, top_branch=None, bottom_branch=bottom_attach)
                    
                    # Get image of path in new decomposition
                    image = B.path_image(path_in_old, low, min(low + eps, high), eta)
                    image = _bridge_append(image, bottom_attach)
                    
                    # Add new branch at end if needed
                    if high > low + eps:
                        image.append(new_branch)
                    
                    eta.set_image(i, image)
                    
                    # Validate
                    if not B_smooth.check_branch_path(image):
                        warnings.warn(
                            f"upfork/local max case created invalid path for branch {i}: {image}",
                            UserWarning,
                            stacklevel=2
                        )
                
                else:
                    # print ("Case 2b: Up fork/down fork")
                
                    if high - low <= 2 * eps:
                        # print("Case 2c: Short upfork/downfork (splitting at midpoint)")
                        # Sliding both endpoints by h/2 would make this branch
                        # horizontal (degenerate) exactly at the midpoint M, since
                        # low + h/2 == high - h/2 == M. Find where each side would
                        # land at M, same walk as the long case but targeting M
                        # from both directions.
                        M = (low + high) / 2.0

                        # Walk up/down in the *original* decomposition all the way to the
                        # slide targets (low+eps / high-eps), not just to M. Since h <= 2*eps
                        # guarantees low+eps >= M >= high-eps, a walk targeting the slide
                        # height necessarily passes through M as well, so the same eta-mapped
                        # image can resolve both the midpoint-split attachment and the final
                        # post-slide attachment. (Walking structurally in B_smooth from B_older
                        # after the split, as was done previously, is unreliable: Branch only
                        # stores outgoing top_branch/bottom_branch pointers, not who else
                        # attaches into it, so that walk can silently stop short.)
                        check_attach_bottom = b.bottom_branch
                        path_up = [check_attach_bottom]

                        while (check_attach_bottom.top_branch is not None and
                                check_attach_bottom.f_high < low + eps):
                            check_attach_bottom = check_attach_bottom.top_branch
                            path_up.append(check_attach_bottom)

                        check_attach_top = b.top_branch
                        path_down = [check_attach_top]

                        while (check_attach_top.bottom_branch is not None and
                                check_attach_top.f_low >= high - eps):
                            check_attach_top = check_attach_top.bottom_branch
                            path_down.append(check_attach_top)

                        path_down = list(reversed(path_down))

                        image_up = eta.get_image(B._index_of_branch(path_up[-1]))
                        bottom_attach = _attach_branch_at_height(image_up, M)
                        final_bottom_attach = _attach_branch_at_height(image_up, low + eps)

                        image_down = eta.get_image(B._index_of_branch(path_down[0]))
                        top_attach = _attach_branch_at_height(image_down, M)
                        final_top_attach = _attach_branch_at_height(image_down, high - eps)

                        if bottom_attach is top_attach:
                            raise ValueError(
                                f"branch {i}: bottom and top attachments coincide at the same "
                                "branch at the midpoint; this degenerate sub-case is not yet handled."
                            )

                        # Whichever attachment is later in B_smooth gets split in two;
                        # the earlier one becomes the shared point for both halves.
                        if B_smooth._index_of_branch(bottom_attach) > B_smooth._index_of_branch(top_attach):
                            B_later, B_older = bottom_attach, top_attach
                        else:
                            B_later, B_older = top_attach, bottom_attach

                        # The pre-split slide targets were resolved from the same original
                        # branch's own image as bottom_attach/top_attach. If that image is a
                        # dead end (e.g. a lone local-min/max branch with nothing past it, as
                        # smB3 is here), the target can come back as B_later itself, which
                        # would be a self-attachment once B_later is split. In that case, the
                        # slide target must instead be found on B_older's side: B_older is
                        # already where both halves attach at M, so fall back to walking
                        # structurally from B_older within B_smooth (same approach as the
                        # eta-based walk, just starting one step further along).
                        if final_bottom_attach is B_later:
                            check = B_older
                            walk = [check]
                            while check.top_branch is not None and check.f_high < low + eps:
                                check = check.top_branch
                                walk.append(check)
                            final_bottom_attach = _attach_branch_at_height(walk, low + eps)
                            assert final_bottom_attach is not B_later, (
                                "final bottom attachment fallback still resolved to B_later"
                            )
                        if final_top_attach is B_later:
                            check = B_older
                            walk = [check]
                            while check.bottom_branch is not None and check.f_low >= high - eps:
                                check = check.bottom_branch
                                walk.append(check)
                            walk = list(reversed(walk))
                            final_top_attach = _attach_branch_at_height(walk, high - eps)
                            assert final_top_attach is not B_later, (
                                "final top attachment fallback still resolved to B_later"
                            )

                        # B_later_low and B_later_high will NOT be attached to each other once
                        # split -- they become two independent branches, each wired to its own
                        # final slide target (final_bottom_attach / final_top_attach), which
                        # may lie on entirely unrelated parts of B_smooth. Any stale eta path
                        # that needs to pass from one split half to the other must instead
                        # route through whatever already connects final_top_attach to
                        # final_bottom_attach elsewhere in the (connected) graph -- found here
                        # via a live graph search rather than assumed.
                        bridge = _find_connecting_path(final_top_attach, final_bottom_attach)

                        B_later_low = B_smooth.insert_before(
                            B_later, f_low=B_later.f_low, f_high=M,
                            bottom_branch=B_later.bottom_branch, top_branch=B_older,
                        )
                        B_later_high = B_smooth.insert_after(
                            B_later_low, f_low=M, f_high=B_later.f_high,
                            bottom_branch=B_older, top_branch=B_later.top_branch,
                        )

                        # Repoint any existing attachment into B_later to whichever half
                        # now contains it; an exact tie at M means it now shares B_older's point.
                        # (list(...) copies since these setters mutate B_later's own lists.)
                        for other in list(B_later.attached_via_top):
                            if other.f_high < M:
                                other.top_branch = B_later_low
                            elif other.f_high > M:
                                other.top_branch = B_later_high
                            else:
                                other.top_branch = B_older
                        for other in list(B_later.attached_via_bottom):
                            if other.f_low < M:
                                other.bottom_branch = B_later_low
                            elif other.f_low > M:
                                other.bottom_branch = B_later_high
                            else:
                                other.bottom_branch = B_older

                        # Fix up any previously-set eta[j] paths that still reference
                        # B_later's key before B_later is removed and its own
                        # top_branch/bottom_branch (needed to resolve entry/exit
                        # heights) get cleared.
                        _repoint_stale_eta_paths(B_later, B_later_low, B_later_high, M, bridge)

                        B_smooth.remove(B_later)

                        # Anything attached in the now-excluded sliver of B_later_high /
                        # B_later_low (the part the slide is moving past) must be
                        # reattached to wherever it now falls along the eta-mapped image --
                        # which may be B_older itself, an intermediate branch, or the
                        # final target.
                        for other in list(B_later_high.attached_via_bottom):
                            if other.f_low <= low + eps:
                                other.bottom_branch = _attach_branch_at_height(image_up, other.f_low)
                        for other in list(B_later_high.attached_via_top):
                            if other.f_high <= low + eps:
                                other.top_branch = _attach_branch_at_height(image_up, other.f_high)
                        for other in list(B_later_low.attached_via_top):
                            if other.f_high >= high - eps:
                                other.top_branch = _attach_branch_at_height(image_down, other.f_high)
                        for other in list(B_later_low.attached_via_bottom):
                            if other.f_low >= high - eps:
                                other.bottom_branch = _attach_branch_at_height(image_down, other.f_low)

                        # Move B_later_high's bottom and B_later_low's top to their final
                        # positions. _attach_branch_at_height already guarantees strict
                        # interior containment against the walked path; also confirm each
                        # branch's own untouched endpoint still leaves a valid (non-degenerate) range.
                        assert final_bottom_attach.f_low < low + eps < final_bottom_attach.f_high, (
                            "final bottom attachment for B_later_high is not strictly interior"
                        )
                        assert low + eps < B_later_high.f_high, (
                            "B_later_high's slid bottom would not be below its own (untouched) top"
                        )
                        B_later_high.bottom_branch = final_bottom_attach
                        B_later_high.f_low = low + eps

                        assert final_top_attach.f_low < high - eps < final_top_attach.f_high, (
                            "final top attachment for B_later_low is not strictly interior"
                        )
                        assert high - eps > B_later_low.f_low, (
                            "B_later_low's slid top would not be above its own (untouched) bottom"
                        )
                        B_later_low.top_branch = final_top_attach
                        B_later_low.f_high = high - eps

                        # eta[i]: this branch vanishes, but its bottom- and top-side
                        # neighbors (path_up/path_down) still need a continuous image
                        # covering [low, high]. path_down's own source entries (e.g.
                        # path_down[0]) may have had B_later in their *stored* eta
                        # image, which _repoint_stale_eta_paths already rewrote above
                        # to correctly route through B_later_low/the bridge/B_later_high
                        # -- so path_image(path_down, ...) already resolves the correct
                        # sub-path. Since both sides converge through the same shared
                        # structure once the branch collapses, merge (not concatenate)
                        # to drop whatever overlap results.
                        image = B.path_image(path_up, low, min(low + eps, high), eta)
                        image = _bridge_append(image, final_bottom_attach)

                        image_top = B.path_image(path_down, max(high - eps, low), high, eta)
                        image_top = _bridge_prepend(image_top, final_top_attach)
                        image = _merge_overlapping(image, image_top)


                        eta.set_image(i, image)

                        if not B_smooth.check_branch_path(image):
                            warnings.warn(
                                f"short upfork/downfork case created invalid path for branch {i}: {image}",
                                UserWarning,
                                stacklevel=2
                            )
                    else:
                        # Long edge case: upfork slides up, downfork slides down
                        # Follow up from bottom attachment until above low+eps
                        check_attach_bottom = b.bottom_branch
                        path_up = [check_attach_bottom]
                    
                        while (check_attach_bottom.top_branch is not None and 
                                check_attach_bottom.f_high < low + eps):
                            check_attach_bottom = check_attach_bottom.top_branch
                            path_up.append(check_attach_bottom)
                    
                        # Follow down from top attachment until below high-eps
                        check_attach_top = b.top_branch
                        path_down = [check_attach_top]
                    
                        while (check_attach_top.bottom_branch is not None and 
                                check_attach_top.f_low >= high - eps):
                            check_attach_top = check_attach_top.bottom_branch
                            path_down.append(check_attach_top)
                    
                        # Reverse path_down since we followed down but need to go up
                        path_down = list(reversed(path_down))
                    
                        # Get attachments for the new middle branch, using the branch in each image
                        # path that actually contains the attachment height, not just its endpoint.
                        last_branch_image_up = eta.get_image(B._index_of_branch(path_up[-1]))
                        bottom_attach = _attach_branch_at_height(last_branch_image_up, low + eps)
                    
                        first_branch_image_down = eta.get_image(B._index_of_branch(path_down[0]))
                        top_attach = _attach_branch_at_height(first_branch_image_down, high - eps)
                    
                        # Create the middle branch with both attachments
                        new_branch = B_smooth.append(low + eps, high - eps, top_branch=top_attach, bottom_branch=bottom_attach)
                    
                        # Get the image of the upfork path (restricted to interval (low, min(low+eps, high)))
                        image_up = B.path_image(path_up, low, min(low + eps, high), eta)
                        image_up = _bridge_append(image_up, bottom_attach)
                    
                        # Get the image of the downfork path (restricted to interval (max(high-eps, low), high))
                        image_down = B.path_image(path_down, max(high - eps, low), high, eta)
                        image_down = _bridge_prepend(image_down, top_attach)
                    
                        # Combine images: upfork path + new middle branch + downfork path
                        image = image_up
                        image.append(new_branch)
                        image.extend(image_down)
                    
                        eta.set_image(i, image)
                    
                        # Validate
                        if not B_smooth.check_branch_path(image):
                            warnings.warn(
                                f"upfork/downfork case created invalid path for branch {i}: {image}",
                                UserWarning,
                                stacklevel=2
                            )
        return B_smooth, eta
    
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


class BranchDecompMap:
    """A map between two branch decompositions which are stored internally.

    Source branches and target image paths are stored by UUID, so the mapping remains stable when either decomposition's linked-list order changes.
    """

    def __init__(self, source: BranchDecomp, target: BranchDecomp):
        if not isinstance(source, BranchDecomp) or not isinstance(target, BranchDecomp):
            raise TypeError("source and target must be BranchDecomp instances")

        self.source = source
        self.target = target
        self.image_paths: dict[uuid.UUID, list[uuid.UUID]] = {}

    def set_image(self, source_branch, target_path, validate: bool = False) -> None:
        """Set a source branch's image as a path in the target decomposition.

        References may be branches, UUIDs, or list-order indices. Internally,
        the mapping is stored entirely as UUIDs. An empty target path is
        permitted for a branch whose image vanishes.
        
        Args:
            source_branch: Reference to a source branch (Branch, UUID, or int index).
            target_path: Path in the target decomposition (list of Branch/UUID/int references).
            validate: Whether to validate that target_path forms a valid upward path.
                     Defaults to False to allow incremental construction.
        """
        source = self.source._resolve_branch_ref(source_branch)
        if source is None:
            raise ValueError("source_branch cannot be None")

        if not target_path:
            self.image_paths[source.key] = []
            return

        target_branches = self.target._resolve_path(target_path)
        if validate and not self.target.check_branch_path(target_branches):
            raise ValueError("target_path is not a valid path in the target decomposition")

        self.image_paths[source.key] = [branch.key for branch in target_branches]

    def get_image_keys(self, source_branch) -> list[uuid.UUID]:
        """Return a copy of a source branch's stored target UUID path."""
        source = self.source._resolve_branch_ref(source_branch)
        if source is None:
            raise ValueError("source_branch cannot be None")
        return list(self.image_paths[source.key])

    def get_image(self, source_branch) -> list[Branch]:
        """Resolve a source branch's image path to target branch objects."""
        return [self.target.get_by_key(key) for key in self.get_image_keys(source_branch)]

    def get_image_indices(self, source_branch) -> list[int]:
        """Return a source branch's image path in the target's current list order."""
        return [
            self.target._index_of_branch(self.target.get_by_key(key))
            for key in self.get_image_keys(source_branch)
        ]

    def __getitem__(self, source_index: int) -> list[int]:
        """Return the image of a source list-order index as target indices."""
        if not isinstance(source_index, int):
            raise TypeError("source index must be an integer")
        return self.get_image_indices(source_index)

    def remove_image(self, source_branch) -> None:
        """Remove the stored image of a source branch, if present."""
        source = self.source._resolve_branch_ref(source_branch)
        if source is None:
            raise ValueError("source_branch cannot be None")
        self.image_paths.pop(source.key, None)


