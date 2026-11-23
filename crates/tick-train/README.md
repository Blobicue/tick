# tick-train

From-scratch ternary QAT: float masters with a derived BitNet
abs-mean ternary view, straight-through-estimator backward, q8
activation STE (the backward trains on exactly the values inference
multiplies), batched steps, and best-val checkpoint snapshots the
stock engine runs directly.

The m41–m43 examples are runnable milestones with printed PASS gates:
memorization, held-out generalization, batched QAT with
checkpoint-on-improve.
