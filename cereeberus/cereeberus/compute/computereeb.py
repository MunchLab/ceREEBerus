from itertools import groupby as _groupby

import numpy as np

from ..reeb.lowerstar import LowerStar
from .unionfind import UnionFind


def is_face(sigma, tau):
    """
    Check if tau is a face of sigma

    Args:
        sigma: A simplicial complex (a list of simplices).
        tau: A simplex (a list of vertices).

    Returns:
        bool: True if tau is a face of sigma, False otherwise.
    """
    return set(tau).issubset(set(sigma))


def get_levelset_components(L):
    """
    Given a list of simplices L representing a level set, compute the connected components. This function is really only helpful inside of computeReeb.

    Args:
        L: A list of simplices (each simplex is a list of vertices).

    Returns:
        dict: A dictionary where keys are representative simplices and values are lists of simplices in the same connected component.
    """

    UF = UnionFind(range(len(L)))
    for i, simplex1 in enumerate(L):
        for j, simplex2 in enumerate(L):
            if i < j:
                # Check if they share a vertex
                if is_face(simplex1, simplex2) or is_face(simplex2, simplex1):
                    UF.union(i, j)

    # Replace indices with simplices
    components_index = UF.components_dict()
    components = {}
    for key in components_index:
        components[tuple(L[key])] = [L[i] for i in components_index[key]]

    return components


def computeReeb(K: LowerStar, verbose=False):
    """Computes the Reeb graph of a Lower Star Simplicial Complex K.

    Args:
        K (LowerStar): A Lower Star Simplicial Complex with assigned filtration values.
        verbose (boolean): Make it True if you want lots of printouts.

    Returns:
        ReebGraph: The computed Reeb graph.

    Example:
        >>> from cereeberus.reeb.LowerStar import LowerStar
        >>> K = LowerStar()
        >>> K.insert([0, 1, 2])
        >>> K.insert([1, 3])
        >>> K.insert([2,3])
        >>> K.assign_filtration([0], 0.0)
        >>> K.assign_filtration([1], 3.0)
        >>> K.assign_filtration([2], 5.0)
        >>> K.assign_filtration([3], 7)
        >>> R = computeReeb(K)
        >>> R.draw()
    """
    from ..reeb.reebgraph import ReebGraph

    funcVals = [(i, K.filtration([i])) for i in K.iter_vertices()]
    funcVals.sort(key=lambda x: x[1])  # Sort by filtration value
    # Group vertices that share the same filtration value into batches.
    # A horizontal edge (both endpoints at the same height) must be processed
    # within one batch so it properly merges its endpoints into a single Reeb node.
    grouped = [
        (filt, list(grp))
        for filt, grp in _groupby(funcVals, key=lambda x: x[1])
    ]

    R = ReebGraph()
    currentLevelSet = []
    half_edge_index = 0
    vert_to_component = {}
    edges_at_prev_level = []

    def _dedup(lst):
        seen = set()
        out = []
        for s in lst:
            key = tuple(sorted(s))
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    for group_idx, (filt, group_verts) in enumerate(grouped):
        now_min = filt
        now_max = (
            grouped[group_idx + 1][0] if group_idx + 1 < len(grouped) else np.inf
        )
        vert_names = [v for v, _ in group_verts]

        if verbose:
            print(f"\n---\n Processing group at func val {filt:.2f}: {vert_names}")

        # Classify all simplex stars for this batch into three groups:
        #
        #   lower_nonhoriz: s_filt == filt AND at least one vertex strictly below filt.
        #       These were added to currentLevelSet by an earlier vertex; remove them now.
        #
        #   horizontal:     s_filt == filt AND ALL vertices are at filt.
        #       These connect same-height vertices in the same batch.
        #       Add them temporarily so they link the endpoints into one component.
        #
        #   upper:          s_filt > filt.
        #       Add persistently; they carry the level set upward to the next critical point.
        #
        # Note: because filt(simplex) = max(vertex filtrations) >= filt for any simplex
        # in the star of a vertex at height filt, s_filt < filt is impossible here.
        all_lower_nonhoriz = []
        all_upper = []
        all_horizontal = []

        for vert in vert_names:
            for s in K.get_star([vert]):
                simplex, s_filt = s[0], s[1]
                if len(simplex) <= 1:
                    continue
                if s_filt > filt:
                    all_upper.append(simplex)
                elif all(K.filtration([u]) == filt for u in simplex):
                    all_horizontal.append(simplex)
                else:
                    all_lower_nonhoriz.append(simplex)

        all_lower_nonhoriz = _dedup(all_lower_nonhoriz)
        all_upper = _dedup(all_upper)
        all_horizontal = _dedup(all_horizontal)

        if verbose:
            print(f"  Lower (non-horiz) simplices: {all_lower_nonhoriz}")
            print(f"  Horizontal simplices: {all_horizontal}")
            print(f"  Upper simplices: {all_upper}")

        # Step 1: Remove the lower non-horizontal simplices from the active level set.
        for s in all_lower_nonhoriz:
            if s in currentLevelSet:
                currentLevelSet.remove(s)

        # Step 2: Add this batch's vertices and its horizontal simplices to the level set.
        for vert in vert_names:
            currentLevelSet.append([vert])
        for s in all_horizontal:
            if s not in currentLevelSet:
                currentLevelSet.append(s)

        if verbose:
            print(f"  Current level set: {currentLevelSet}")

        # Step 3: Compute connected components; create one Reeb node per component.
        components_at_level = get_levelset_components(currentLevelSet)

        if verbose:
            print(f"  Level set components:")
            for comp in components_at_level.values():
                print(f"    {comp}")

        verts_at_level = []
        for rep, comp in components_at_level.items():
            nextNodeName = R.get_next_vert_name()
            R.add_node(nextNodeName, now_min)
            vert_to_component[nextNodeName] = comp
            verts_at_level.append(nextNodeName)

            # Connect incoming half-edge sentinels whose component contains a face
            # of a simplex in this component.
            for e in edges_at_prev_level:
                prev_comp = vert_to_component[e]
                if any(
                    is_face(prev_simp, simp)
                    for simp in comp
                    for prev_simp in prev_comp
                ):
                    R.add_edge(e, nextNodeName)

        # Step 4: Remove vertices and horizontal simplices – they live only at this exact height.
        for vert in vert_names:
            if [vert] in currentLevelSet:
                currentLevelSet.remove([vert])
        for s in all_horizontal:
            if s in currentLevelSet:
                currentLevelSet.remove(s)

        # Step 5: Add upper-star simplices to carry the level set forward.
        for s in all_upper:
            if s not in currentLevelSet:
                currentLevelSet.append(s)

        if verbose:
            print(f"\n  Updated level set: {currentLevelSet}")

        # Step 6: Compute components above this level; create half-edge sentinel nodes
        # at height (now_min + now_max) / 2 and connect them downward to the Reeb nodes
        # created in Step 3.
        components_above = get_levelset_components(currentLevelSet)

        if verbose:
            print(f"  Level set components above:")
            for comp in components_above.values():
                print(f"    {comp}")

        edges_at_prev_level = []
        for comp in components_above.values():
            e_name = "e_" + str(half_edge_index)
            R.add_node(e_name, (now_min + now_max) / 2)
            vert_to_component[e_name] = comp
            half_edge_index += 1
            edges_at_prev_level.append(e_name)

            # Connect downward to Reeb nodes at this level.
            for v in verts_at_level:
                prev_comp = vert_to_component[v]
                if any(
                    is_face(simp, prev_simp)
                    for simp in comp
                    for prev_simp in prev_comp
                ):
                    R.add_edge(v, e_name)

    return R

