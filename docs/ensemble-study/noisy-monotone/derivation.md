# Exact empirical minimizers, allowed mistakes and target Brier

For a binary function f and an observation (x,y), the Hamming loss at x is either f(x) or 1−f(x). For any f and g,

```math
L(f\wedge g)+L(f\vee g)=L(f)+L(g).
```

The identity holds separately on every row, even when duplicates carry contradictory labels. If f and g minimize empirical loss over a finite function class closed under meet and join, each term on the left is at least the empirical minimum. Both must attain that minimum. Thus the minimizers themselves form a lattice with feasible least and greatest elements. Their midpoint predicts 0.5 exactly in the ambiguous region, with common binary Brier D/4 against each empirical minimizer under a target law assigning mass D to that region.

On the full Boolean cube the empirical fit is a directed s–t minimum cut. Send positive-observation counts from the source to each input and negative counts from each input to the sink. Put an edge of effectively infinite capacity from a point to each featurewise immediate successor. A source-side node is labeled one, so a finite cut cannot put a positive point below a negative one. Force the all-zero and all-one assignments at the endpoints. The minimum cut capacity is precisely the smallest count of training mistakes. In a maximum-flow residual graph, the nodes reachable from the source must lie on the source side of every minimum cut. Its maximal source side excludes exactly the nodes from which the sink remains reachable. These give the two bounding ERM predictions.

An error budget B is different. If L(f), L(g) ≤ B, modularity bounds the *sum* of losses at their meet and join by 2B. One term can exceed B. For any finite candidate matrix H on target coordinates with weights q, the exact minimax Brier is the squared radius of the smallest enclosing ball under the weighted Euclidean norm. Given a distribution π over candidates, its weighted predictive variance is a lower bound on minimax risk. The maximum empirical squared distance from any reported center to a candidate is an upper bound. Their gap quantifies numerical error when the feasible set lacks bounding members.

Three independent-coordinate singleton queries make the failure explicit. All three observed labels are zero. Under at most one incorrect label, the projected truth possibilities are 000, 100, 010 and 001; each has a nonconstant monotone extension with the required anchors. The upper vector 111 is infeasible. By coordinate symmetry an optimal center predicts a common q on the three equally weighted target points. Worst squared error is max(q², q²−2q/3+1/3). It is minimized at q=1/3 with risk 2/9. The midpoint of pointwise bounds has risk 1/4. Exact empirical minimization uses q=0 on these points and incurs 1/3 if a generating positive label was flipped.
