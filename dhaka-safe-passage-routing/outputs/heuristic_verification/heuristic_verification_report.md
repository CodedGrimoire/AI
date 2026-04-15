# Heuristic Admissibility and Consistency Evaluation

## Experimental Setup
- Graph: Dhaka road network (projected cache)
- Graph size: 103386 nodes, 253424 edges
- Sampled nodes per goal: 3000
- Sampled directed edges: 8000
- Number of goal nodes: 1
- Goal node IDs: 4594175828
- Random seed: 2026
- Edge-cost model used:
  - `cost(u,v)=length(u,v)*max(min_multiplier,1+alpha*T+beta*A+gamma*B-lambda*S)`
  - Weights: `{'alpha_traffic': 0.9, 'beta_accident': 1.1, 'gamma_bumpiness': 0.6, 'lambda_safety': 0.8, 'min_multiplier': 0.05}`
- Heuristic formula used:
  - `h(n)=d(n,g)*max(min_factor,1+alpha*T(n)+beta*A(n)+gamma*B(n)-lambda*S(n))`
  - Weights: `{'alpha_traffic': 0.7, 'beta_accident': 0.9, 'gamma_bumpiness': 0.5, 'lambda_safety': 0.8, 'min_factor': 0.05}`

## Admissibility Results
- Total tested node-goal pairs: 3000
- Skipped unreachable pairs: 0
- Number of violations: 0
- Violation rate: 0.000000
- Max overestimation: 0.000000
- Avg overestimation among violations: 0.000000

## Consistency Results
- Total tested edges: 8000
- Number of violations: 3831
- Violation rate: 0.478875
- Max positive residual: 71752.945194
- Avg positive residual among violations: 7348.391085

## Interpretation
- The heuristic appears empirically admissible on the tested samples.
- The heuristic is not empirically consistent on the tested edges.

## Example Violating Cases
### Top 10 Admissibility Violations
No violations found.

### Top 10 Consistency Violations
| u | v | goal_id | h_u | edge_cost | h_v | residual |
| --- | --- | --- | --- | --- | --- | --- |
| 9952570158.000000 | 9952570136.000000 | 4594175828.000000 | 326111.064163 | 2678.205963 | 251679.913006 | 71752.945194 |
| 4632141202.000000 | 4632141205.000000 | 4594175828.000000 | 299970.554909 | 238.690316 | 230137.926395 | 69593.938198 |
| 5665591067.000000 | 5665591059.000000 | 4594175828.000000 | 341037.373353 | 810.122305 | 272517.781052 | 67709.469996 |
| 7113600019.000000 | 7113599998.000000 | 4594175828.000000 | 323238.806295 | 994.262588 | 257357.137304 | 64887.406402 |
| 4869512558.000000 | 4916747070.000000 | 4594175828.000000 | 243233.621962 | 2871.273320 | 182250.604989 | 58111.743654 |
| 9075629176.000000 | 11957603524.000000 | 4594175828.000000 | 279825.016736 | 1347.062523 | 221924.731459 | 56553.222754 |
| 11981254779.000000 | 11981254773.000000 | 4594175828.000000 | 257745.196111 | 156.209567 | 203080.860795 | 54508.125749 |
| 10047712220.000000 | 10047712221.000000 | 4594175828.000000 | 219738.567819 | 9.608319 | 166430.714549 | 53298.244951 |
| 4869533324.000000 | 6054492595.000000 | 4594175828.000000 | 227665.170094 | 334.264375 | 174044.429047 | 53286.476673 |
| 12338848285.000000 | 9425307387.000000 | 4594175828.000000 | 347327.759008 | 856.525085 | 296390.881942 | 50080.351982 |

## Generated Artifacts
- `admissibility_results.csv`
- `consistency_results.csv`
- `admissibility_summary_by_goal.csv`
- `heuristic_verification_report.md`
- `admissibility_delta_histogram.png`
- `consistency_residual_histogram.png`
- `heuristic_violation_rates.png`

## Conclusion
Empirical checks found violations; the heuristic behaves as a non-admissible and/or non-consistent heuristic in this setup, so strict optimality guarantees are not supported.
