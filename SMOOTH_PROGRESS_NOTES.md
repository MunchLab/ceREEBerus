# Progress notes: `BranchDecomp.smooth()` implementation

Working file: `cereeberus/cereeberus/reeb/branchdecomp.py`
Test notebook: `doc_source/experimental/sandbox_decomposition.ipynb` (the `T` example, cell calling `T.smooth(...)`)
Test suite: `tests/test_branchdecomp.py` (`/opt/anaconda3/bin/python3 -m pytest tests/test_branchdecomp.py -q` -- see note in item 6 about why not `make tests`/repo `.venv`)

## Session 2 (2026-08-20): backward pointers + real bug fixes for Case 2c (vanishing branch)

Key conceptual clarification from user (important -- read before touching this
code again): when a branch (`B_later`) gets split into `B_later_low`/
`B_later_high` because some *other* branch's collapse-at-midpoint needs to
land inside it, **the two halves are not attached to each other and should
not be**. They become two independent branches, each wired to its own,
independently-resolved final slide target (`final_bottom_attach` for
`B_later_high`, `final_top_attach` for `B_later_low`). Any code that needs to
route "from one half to the other" must find whatever *already* connects
`final_top_attach` to `final_bottom_attach` elsewhere in the live graph (since
the Reeb graph is connected, some route must exist) -- never assume a direct
or `B_older`-shaped connection.

1. **Added backward-pointer bookkeeping to `Branch`**: `top_branch`/
   `bottom_branch` are now properties (backed by `_top_branch`/
   `_bottom_branch`) whose setters maintain `attached_via_top`/
   `attached_via_bottom` lists (branches that reference *this* branch as their
   top/bottom attachment). `BranchDecomp.remove()` clears a removed node's own
   pointers via the setters so it doesn't linger in another branch's lists.
   The Case 2c repoint-scan loops use these lists directly instead of
   scanning all of `B_smooth`.

2. **Added `_find_connecting_path(start, end)`** (a BFS inside `smooth()`,
   using `top_branch`/`bottom_branch`/`attached_via_top`/`attached_via_bottom`
   as edges) to find the live path connecting any two branches in
   `B_smooth`. This replaced an earlier, WRONG attempt that assumed the
   connector was always `B_older` (only true by coincidence in degenerate
   test cases -- broke on a genuine non-degenerate short-branch example).
   Used as: `bridge = _find_connecting_path(final_top_attach,
   final_bottom_attach)` (ascending, since high-eps <= low+eps always for
   Case 2c), then `[low_half] + bridge + [high_half]` when a stale eta path
   needs to cross the split.

3. **Fixed a real off-by-one bug in `find_subpath`**: when a query's upper
   bound `b` lands *exactly* on an attachment height owned by the "next"
   branch (the `elif a < attachment_value` / second-owns-bottom branch), the
   old code returned early without ever including that next branch. Changed
   `if b <= attachment_value: return` to `if b < attachment_value: return` in
   that branch so the boundary-owning branch gets included (via the next
   loop iteration or the final `last_branch` fallback).

4. **Fixed `eta[i]`'s own assembly for Case 2c** (the vanishing branch's
   image): previously manually re-inserted `[B_later_high, B_later_low]`
   (wrong order too) into the middle of the path. Since `path_down`'s own
   stored eta entries were *already* correctly fixed by
   `_repoint_stale_eta_paths` (using the new bridge), manually re-inserting
   the split halves just duplicated content. Removed that, and instead
   **merge** (not concatenate) `path_up`'s image with `path_down`'s image via
   a new `_merge_overlapping(first, second)` helper: since both sides
   converge through the same shared live structure once the branch
   collapses, `second` may re-derive a prefix that `first` already ends with
   -- find where `first[-1]` reappears in `second` and drop everything up
   through that point before joining.

5. **Verified**: both of the user's originally-reported failing cases
   (`T.smooth(1.5)` crash, and `T.smooth(2)`'s silently-invalid `eta[4]`) are
   now fixed and produce valid, non-redundant paths (`eta[4] = [0, 1, 4]` for
   both, once indices are aligned to that eps). `tests/test_branchdecomp.py`:
   21 passed, 1 pre-existing unrelated failure (see below).

6. **Fixed the remaining edge case**: `find_subpath` now tracks whether it
   has "started" including branches. Before, each pair's inclusion decision
   was recomputed independently from the original query bound `a`, which
   could skip a zero-width pass-through branch (one entered and exited at
   the *same* height, e.g. a degree-4+ shared vertex) sitting between two
   branches that otherwise both got included -- producing an internally
   disconnected (invalid) result. Fix: once the first branch is included,
   every subsequent "first" in the loop is a connector already committed to
   and must be appended unconditionally (regardless of whether it owns any
   of `[a, b]` itself); only the *initial* inclusion decision and the final
   `b`-vs-`return` check still depend on ownership. All three test cases
   (`eps=1` exact-coincidence, `eps=1.5`, `eps=2`) now produce valid,
   non-redundant `eta[4]` paths.

7. **Environment note**: the repo's `.venv` is Python 3.9 and can't run this
   file (uses `X | Y` union type annotations needing 3.10+). Use
   `/opt/anaconda3/bin/python3` (3.11) for ad hoc debug scripts and for
   `pytest tests/test_branchdecomp.py`. `make tests` / `python -m unittest`
   fails in this environment on unrelated numpy/sklearn/pyarrow ABI errors
   (`numpy.dtype size changed`, `_ARRAY_API not found`) when importing
   `cereeberus` at all -- not related to this work.

## Known pre-existing (unrelated) test failure

`test_branch_decomp_map_stores_uuid_paths` in `tests/test_branchdecomp.py` --
confirmed pre-existing/unrelated to all the smoothing work above (predates
this session and the previous one).

## Debug script used this session (recreate if needed)

`/tmp/debug_smooth7.py` -- tests three cases: `eps=1` ("exact 2eps", still
broken per item 6), `eps=2` (notebook's original example, now fixed), and
`eps=1.5` (non-degenerate short case, now fixed). All use the same `T`:

```python
T = branch.BranchDecomp()
T.append(-1,8); T.append(2,6,bottom_branch=0); T.append(-2,4,top_branch=1)
T.append(0,9); T.append(3,5,bottom_branch=2,top_branch=3)
```

Run with `/opt/anaconda3/bin/python3 /tmp/debug_smooth7.py 2>&1 | tail -40`
(ignore the numpy/pyarrow/pandas import noise at the top -- unrelated).

## Next steps when resuming

All originally-reported bugs plus the follow-on edge cases found while fixing
them are resolved as of this session; `tests/test_branchdecomp.py` is clean
except the pre-existing unrelated failure. Suggested next steps if further
work is wanted:
1. Broader test sweep: more branches, deeper nesting of Case 2c splits (a
   split branch itself later needing to split again), to make sure the
   backward-pointer/bridge/`_find_connecting_path` machinery generalizes.
2. Revisit the pre-existing `test_branch_decomp_map_stores_uuid_paths`
   failure if it becomes relevant (still unaddressed, explicitly deferred).

See terminal history for the exact commands; in summary:

- Basic short-branch split+slide (h == 2*eps exactly, no-op slide):
  `T.append(-1,8); T.append(2,6,bottom_branch=0); T.append(-2,4,top_branch=1);
  T.append(0,9); T.append(3,4,bottom_branch=2,top_branch=3); T.smooth(0.5)`
- Sliver-repoint test (h < 2*eps, with another branch attached in the sliver):
  `T.append(-1,8); T.append(2,6,bottom_branch=0); T.append(-2,4,top_branch=1);
  T.append(0,9); T.append(2.35,8,bottom_branch=3); T.append(2.5,3.3,bottom_branch=2,top_branch=3);
  T.smooth(0.5)` — confirms branch 4 gets correctly reattached from the split branch to
  `B_older` during the slide.

1. **Fixed attach-branch selection bug** in the long-branch (Case 1b / 2a / 2b) code: was
   picking the attach target by blindly indexing `[0]`/`[-1]` into an image path instead of
   finding the branch where the target height is actually interior. Added helper
   `_attach_branch_at_height(path, height)` inside `smooth()` that walks a path's
   attachment values and deterministically resolves ties using the construction-time
   invariant (an attachment point is always guaranteed strictly interior to exactly one
   side of a transition — never both, never neither). Raises `ValueError` if a height
   isn't interior to anything in the path (signals a genuine degenerate/local-min-max
   height).

2. **Relaxed `check_branch_path`** to allow **non-decreasing** (not strictly increasing)
   consecutive attachment heights. Equal consecutive heights are valid — they mean two
   branches share a single vertex (a degree-4+ point), which is a legitimate (if
   non-generic) Reeb graph configuration. Only actual decreases are invalid now.

3. **Added `_bridge_append` / `_bridge_prepend` helpers** in `smooth()`: `find_subpath`'s
   slicing convention can stop a path slice one hop short of the branch where a height is
   actually interior (an ownership-convention artifact, different from what attaching a
   new endpoint needs). These helpers detect that one-hop gap and bridge it by inserting
   the missing adjacent branch. Used when assembling `image`/`image_up`/`image_down` in
   Case 1b/2a/2b.

4. **Implemented the short-branch case** (`high - low <= 2*eps`, Case "2c"), which was
   previously just `eta.set_image(i, [])`. The math (all confirmed working via manual
   smoke tests):
   - Conceptually: smoothing by `eps` on both ends would make the branch horizontal
     (degenerate) at the midpoint `M = (low+high)/2`, since `low+h/2 == high-h/2 == M`.
   - Find `bottom_attach` (walk up from `b.bottom_branch`'s image, targeting `M`) and
     `top_attach` (walk down from `b.top_branch`'s image, targeting `M`), using
     `_attach_branch_at_height`.
   - Whichever of `{bottom_attach, top_attach}` has the **later** index in `B_smooth` is
     `B_later`; the other is `B_older`.
   - **Split `B_later` at `M`** into `B_later_low` (keeps `B_later`'s original bottom
     attachment, new top attaches to `B_older`) and `B_later_high` (keeps `B_later`'s
     original top attachment, new bottom attaches to `B_older`), inserted at `B_later`'s
     old position via `insert_before`/`insert_after`, then `remove(B_later)`.
   - **Repoint scan**: before removing `B_later`, scan all of `B_smooth` for any other
     branch whose `top_branch`/`bottom_branch` pointed at `B_later`, and repoint to
     whichever half (`B_later_low`/`B_later_high`) now contains that height — or to
     `B_older` if the height happens to land exactly on `M` (a further tie).
   - **Remaining slide**: since splitting at `M` only accounts for `h/2` of the `eps`
     smoothing, still need to slide `B_later_high`'s bottom up to `low+eps` and
     `B_later_low`'s top down to `high-eps`, walking from `B_older` (already inside
     `B_smooth`, no `eta` lookup needed) — same walk pattern as the long case
     (`path_up_slide`, `path_down_slide`).
   - **"Now-excluded sliver" bookkeeping**: anything already reattached to
     `B_later_low`/`B_later_high` during the split step, at a height that falls in the
     region the slide is shrinking past (`(high-eps, M)` for `B_later_low`, `(M, low+eps)`
     for `B_later_high`), must be re-repointed to wherever it now falls along
     `path_down_slide`/`path_up_slide` (via `_attach_branch_at_height` again). **Tested
     and confirmed working** with a constructed example (see terminal history: branch
     appended as `T.append(2.35, 8, bottom_branch=3)` correctly got reattached from the
     split branch to `B_older` after the slide).
   - Added explicit assertions (interior-invariant checks) when mutating attachment
     pointers directly, since we bypass `append()`'s automatic validation.
   - `eta.set_image(i, [])` is still just a placeholder for this case — **not yet fixed,
     this is where we left off.**

5. **Added tests** for the relaxed `check_branch_path` / tie-handling behavior (mentioned
   earlier in conversation — check `tests/test_branchdecomp.py` diff for exact additions,
   should be near tests for `check_branch_path`/`find_subpath`).

## Known pre-existing (unrelated) test failure

`test_branch_decomp_map_stores_uuid_paths` in `tests/test_branchdecomp.py` fails on `make
tests` — confirmed this is **pre-existing, unrelated** to all the above work (need to
double check against a clean checkout if this comes up again, but user was already told
"we'll come back to that later").

## In progress / where we stopped: what should `eta[i]` be for the short-branch case?

Conceptual model agreed so far: under eps-smoothing, a branch too short to survive
(`h <= 2eps`) has its *entire* original interval `[low, high]` collapse onto a
(possibly zero-width) overlap region around `M`, since `low+eps` and `high-eps` cross.

User's proposed construction (their exact wording): concatenate, low-to-high:

1. **First**: "the path (reversed) following where the top of `B_later_low` slid
   down" — I believe this is exactly `path_down_slide` as already computed (ascending,
   from `high-eps` up to `B_older`).
2. **Next**: "the path we had found while we were getting the branch to horizontal in
   the first place (so the bottom of the branch sliding up followed by the top of the
   branch sliding down but reversed)" — **ambiguous, this is where we stopped**. Two
   possible readings:
   - (a) The *full* low-to-high path from the split step:
     `B.path_image(path_up, low, M, eta)` + `B.path_image(path_down, M, high, eta)`
     (spanning the entire original `[low, high]`, bridged with `_bridge_append`/
     `_bridge_prepend` similarly to the long case, and substituting `B_later_low`/
     `B_later_high` for any stale reference to the now-removed `B_later`).
   - (b) Just the small "bridge" at `M` — i.e. `bottom_attach`/`top_attach` themselves
     (with stale `B_later` substituted for the correct split half), NOT a path spanning
     the whole `[low,high]`.
   - **Problem flagged, not yet resolved**: reading (a) would make piece 2 span
     `[low, high]`, which conflicts with piece 1 (ends around `high-eps`/`M`) and piece 3
     (starts at `M`) — concatenating pieces 1→2→3 in that order would require going
     *backward* in height between piece 1 and piece 2. Waiting on user's clarification
     of which reading (or a third option) is correct.
3. **Third**: "the path following where the bottom of `B_later_high` slid up" — I believe
   this is exactly `path_up_slide` (ascending, from `B_older` up to `low+eps`).

**Next steps when resuming:**

1. Get clarification on piece 2's exact definition (see ambiguity above) — likely need a
   concrete tiny example worked by hand together.
2. Implement whatever the resolved `eta[i]` construction is, watch for the stale
   `B_later` substitution (replace with `B_later_low`/`B_later_high` as appropriate
   depending on which one `B_later` was).
3. Still-deferred issue (raised earlier, explicitly set aside): other, previously
   processed branches' **stored `eta` image paths** may already contain `B_later`'s key
   before it gets removed from `B_smooth`. `eta.get_image()` does `by_key[key]` lookups,
   so those entries will raise `KeyError` once something tries to resolve them. This
   needs a fix-up pass (distinct from the structural repoint-scan already implemented,
   which only fixes `Branch.top_branch`/`bottom_branch` object pointers, not `eta`'s
   stored UUID paths).
4. Re-run `T.smooth(0.5)` on the sandbox notebook's `T` example plus the constructed
   test cases in terminal history to confirm.
5. Run `make tests` again once done (ignore the pre-existing unrelated failure noted
   above unless asked to fix it).

## Useful one-off test snippets (already run, worked)

See terminal history for the exact commands; in summary:

- Basic short-branch split+slide (h == 2*eps exactly, no-op slide):
  `T.append(-1,8); T.append(2,6,bottom_branch=0); T.append(-2,4,top_branch=1);
  T.append(0,9); T.append(3,4,bottom_branch=2,top_branch=3); T.smooth(0.5)`
- Sliver-repoint test (h < 2*eps, with another branch attached in the sliver):
  `T.append(-1,8); T.append(2,6,bottom_branch=0); T.append(-2,4,top_branch=1);
  T.append(0,9); T.append(2.35,8,bottom_branch=3); T.append(2.5,3.3,bottom_branch=2,top_branch=3);
  T.smooth(0.5)` — confirms branch 4 gets correctly reattached from the split branch to
  `B_older` during the slide.
