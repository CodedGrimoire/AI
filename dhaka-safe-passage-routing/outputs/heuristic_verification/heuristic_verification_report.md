# Heuristic Admissibility and Consistency Evaluation

## Experimental Setup
- Graph: Dhaka road network (projected cache)
- Graph size: 103386 nodes, 253424 edges
- Sampled nodes per goal: 3000
- Sampled directed edges: 8000
- Number of goal nodes: 5
- Goal node IDs: 5564620678, 4823668170, 7858285282, 9805619703, 6646102406
- Random seed: 2026
- Edge-cost model used:
  - `cost(u,v)=length(u,v)*max(min_multiplier,1+alpha*T+beta*A+gamma*B-lambda*S)`
  - Weights: `{'alpha_traffic': 0.9, 'beta_accident': 1.1, 'gamma_bumpiness': 0.6, 'lambda_safety': 0.8, 'min_multiplier': 0.05}`
- Heuristic formula used:
  - `h(n)=d(n,g)*max(min_factor,1+alpha*T(n)+beta*A(n)+gamma*B(n)-lambda*S(n))`
  - Weights: `{'alpha_traffic': 0.7, 'beta_accident': 0.9, 'gamma_bumpiness': 0.5, 'lambda_safety': 0.8, 'min_factor': 0.05}`

## Admissibility Results
- Total tested node-goal pairs: 15000
- Skipped unreachable pairs: 0
- Number of violations: 0
- Violation rate: 0.000000
- Max overestimation: 0.000000
- Avg overestimation among violations: 0.000000

## Consistency Results
- Total tested edges: 8000
- Number of violations: 3511
- Violation rate: 0.438875
- Max positive residual: 42737.249877
- Avg positive residual among violations: 2364.792744

## Interpretation
- The heuristic appears empirically admissible on the tested samples.
- The heuristic is not empirically consistent on the tested edges.

## Example Violating Cases
### Top 10 Admissibility Violations
No violations found.

### Top 10 Consistency Violations
| u | v | goal_id | h_u | edge_cost | h_v | residual |
| --- | --- | --- | --- | --- | --- | --- |
| 9952570158.000000 | 9952570136.000000 | 5564620678.000000 | 196273.910705 | 2678.205963 | 150858.454865 | 42737.249877 |
| 4632141202.000000 | 4632141205.000000 | 5564620678.000000 | 164505.835688 | 238.690316 | 126286.080538 | 37981.064833 |
| 5665591067.000000 | 5665591059.000000 | 5564620678.000000 | 191104.313947 | 810.122305 | 152520.486021 | 37773.705621 |
| 7113600019.000000 | 7113599998.000000 | 5564620678.000000 | 168722.682529 | 994.262588 | 134045.065929 | 33683.354012 |
| 8794806573.000000 | 8794806580.000000 | 5564620678.000000 | 143328.691889 | 416.031763 | 110003.405124 | 32909.255002 |
| 11981254779.000000 | 11981254773.000000 | 5564620678.000000 | 150343.795040 | 156.209567 | 118468.273665 | 31719.311808 |
| 10989769264.000000 | 10777324888.000000 | 5564620678.000000 | 158691.522659 | 1134.546075 | 127664.571780 | 29892.404804 |
| 10047712220.000000 | 10047712221.000000 | 5564620678.000000 | 119610.199207 | 9.608319 | 90591.667985 | 29008.922903 |
| 9075629176.000000 | 11957603524.000000 | 5564620678.000000 | 132111.731589 | 1347.062523 | 104290.889688 | 26473.779378 |
| 9164622617.000000 | 9164622620.000000 | 5564620678.000000 | 98176.131987 | 193.779772 | 74022.831049 | 23959.521166 |

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
