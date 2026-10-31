# Same-string segmentation invariance: exact synthetic check

## Purpose and scope

This CPU-only finite example checks the estimator logic before any MT RL work. It is not evidence for or against the fertility hypothesis, and it does not model NLLB's autoregressive token states.

The policy samples one of three strings, `s ∈ {A, B, C}`, from a categorical distribution `pθ(s)`. It then samples a path `z` conditional on that string using `qφ(z|s)`. Strings A and C have short and long paths; B has one fixed path. Reward depends only on the string: `R(A)=1`, `R(B)=0.2`, and `R(C)=-0.7`.

The exact behavior probability is

```text
Pθ,φ(s,z) = pθ(s) qφ(z|s)
```

and the logged behavior score is the derivative of this joint probability. Since `qφ` is conditional and normalized, summing over paths recovers `pθ(s)` exactly.

## What the enumeration establishes

For string-only reward and ordinary response-mean REINFORCE,

```text
E[s,z] [ R(s) ∇θ log Pθ,φ(s,z) ]
= E[s] [ R(s) ∇θ log pθ(s) ]
= ∇θ E[s][R(s)]
```

The `φ` component is zero: conditional on any string, the expected score of `qφ(z|s)` is zero. Therefore changing the segmentation sampler alone cannot change the expected ordinary policy gradient for the string policy.

The default exact run has total joint probability `1.0`. Its exact string gradient is `(0.319126, -0.006511, -0.312615)` for all tested segmentation probabilities. At `q(short|s)=0.25`, exact response-mean `φ` gradient is `0`; the same remains true at `q=0.75`.

## Where estimator changes enter

These are distinct effects and should not be conflated:

1. **Response-mean reduction:** for a batch of `B` responses, compute each sequence's score contribution by summing its valid token-action scores (times its sequence reward/advantage), then average those sequence contributions over the `B` responses. There is no per-response token-count denominator. This is the ordinary estimator in the invariance derivation, so changing only `qφ(z|s)` leaves its expected string-policy gradient unchanged.
2. **Fixed realized batch:** dividing the sum of response score contributions by response count or by total token count differs by the single positive scalar `B / ΣᵢLᵢ`. The gradients are exactly collinear for that batch; only the step size changes.
3. **Expected global token mean across random batches:** the denominator `ΣᵢLᵢ` varies with sampled paths and correlates with their scores and rewards. It therefore changes expected sample weighting. In the default batch-size-2 example, the exact θ gradient is `(0.104144, -0.003355, -0.100789)` at `q(short)=0.25` and `(0.184199, -0.017408, -0.166791)` at `q(short)=0.75`; the exact φ gradient is nonzero (`0.011980` and `0.019061`). These are objective/estimator effects. In this example, the θ direction shift is small (cosine `0.999945` versus ordinary response mean), despite the large scale difference.
4. **Token mean within each response:** divide each response's summed token-action contribution by that response's own valid token count, then average responses. Its denominator depends directly on the sampled segmentation. This is distinct from response-mean reduction. Its default exact θ gradient is `(0.122495, -0.010108, -0.112388)` and its φ gradient is `0.044691`.
5. **Position weighting:** the harness distributes the string score evenly over the sampled path positions, then weights position `t` by `γᵗ`. At `γ=1`, the pieces sum to the ordinary score and reproduce its gradient. At `γ=0.5`, this deliberately changes the estimator; the default θ gradient is `(0.176886, -0.005644, -0.171242)`. This equal-share construction is only a controlled estimator example. It is not the score of `pθ(s)qφ(z|s)` unless a model's actual autoregressive factorization supplies that per-position decomposition.

An alternate estimator may be useful to study as an estimator intervention, but a fertility change under it would not by itself demonstrate improved translation competence or an isolated effect of segmentation.

## Monte Carlo check

Exact enumeration of every ordered batch of size two is compared with 200,000 sampled batches. The Monte Carlo response-mean θ estimate is `(0.319331, -0.006679, -0.312651)` with φ estimate `0.000156`; the global-token-mean estimates are `(0.104187, -0.003388, -0.100799)` and `0.011978`. They agree with their corresponding exact expectations within Monte Carlo error. The tests also verify joint log probabilities and scores outcome by outcome.

Run the harness with:

```bash
python src/exact_same_string_harness.py
PYTHONPATH=src python -m unittest discover -s tests -p 'test_synthetic_validation.py'
```

## Implication for an NLLB intervention

If NLLB first samples a canonical string and a separate process then chooses an alternate segmentation, that sampled joint event must be scored as `log pθ(string) + log qφ(path|string)`. NLLB's raw probability of emitting the alternate token path is not the behavior probability for that two-stage sampler. Under a string-only reward, the exact same-string construction leaves no mechanism for `qφ` to change the ordinary expected gradient of the string policy. Token-mean or position-dependent weights can introduce a path-length or position-dependent estimator effect, which needs to be named and evaluated as such.

If alternate token paths instead run through NLLB's own autoregressive states, the conditional future-token distributions can change. That is a different intervention on the decoder's represented policy and potentially its competence; its causal claim would be a total path-policy effect, not an isolated same-string segmentation effect.

**Decision:** keep MT RL paused. The harness validates the finite sampling and estimator identities; it does not provide the missing mechanism for a competence-preserving segmentation intervention. Any proposed synthetic or model intervention should state that mechanism and estimand before returning to NLLB.
