# Prospective safe-pressure selection

Status: design frozen before selection of a fresh confirmation dataset. The
machine-readable source of truth is
`results/safe_pressure_protocol.json`. The exact future data, reward, document,
segment, and system lists must be recorded in a separately hashed manifest
before any human outcome from that pool is read.

## Question

Can a small calibration audit choose how much a learned machine-translation
reward should be optimized, reducing held-out human-quality regret relative to
fixed optimization pressure and best-of-N?

The completed sign-screen experiment is development evidence. It showed why a
binary optimize-or-keep decision is inadequate: low-pressure optimization was
often beneficial even when the initial gradient sign advised keeping the
uniform policy. This study instead treats optimization pressure as the action.

## Decision problem

For each independently trained reward checkpoint, the procedure chooses one
checkpoint from a frozen grid of normalized policy displacements. Displacement
is mean KL from the uniform policy divided by the maximum KL possible for the
number of actions. Zero pressure therefore remains an explicit decision.

The policy is the existing shared source--candidate-text scorer. It receives no
system identity. All trajectories and their artifact hashes are fixed before
calibration human labels are opened. Calibration decisions are then fixed and
hashed before evaluation human labels are opened.

## Primary selector

The calibration sample contains one frozen segment from each frozen calibration
document. Human scores are standardized within segment so different annotation
scales do not determine the decision. For each jointly reachable pressure, the
selector computes expected standardized human quality under the frozen policy.
It chooses the pressure with the largest mean improvement over uniform, with
ties going to lower pressure. Uniform is always available.

The selector uses bounded calibration feedback. It does not use evaluation
documents, held-out systems, reward-training outcomes from the fresh pool, or
human labels for checkpoint stopping.

## Diagnostics

Before human labels are opened, the study records reward concentration on a
single system, between-system score separation, initial gradient norm, initial
concentration rate, the top-two reward gap, and disagreement across reward
seeds. These are prespecified mechanism diagnostics in this protocol version.
They cannot replace the primary selector after evaluation outcomes are seen.

## Baselines and endpoint

Baselines retain the initial policy, optimize to fixed pressure, optimize to the
largest jointly reachable pressure, and use a small best-of-N calibration
audit. An evaluation-only human oracle supplies the ceiling.

The primary endpoint is held-out decision regret: the difference between the
oracle pressure's human quality and the selected pressure's human quality.
Unreachable checkpoints are reported and never imputed. The primary selector
must beat every fixed-pressure baseline and best-of-N in mean regret, with its
interval excluding zero against the strongest non-oracle baseline, to satisfy
the frozen success criterion.

## Fresh-data boundary

All WMT pools already examined in this project are excluded. An outcome-blind
audit must find a publicly releasable candidate pool with document identifiers,
candidate-level human evaluation, disjoint train and held-out machine systems,
and rectangular candidate coverage. If none exists, the prospective study
stops as infeasible rather than reusing a viewed endpoint as confirmation.

### Outcome-blind candidate audit

The first public candidates are the WMT24 English--German and English--Spanish
MQM pools. The official WMT24 systems repository contains source, reference,
and hypothesis XML for both directions, while the Google MQM repository lists
separate human-evaluation TSVs for them. The human TSV contents have not been
opened for this audit.

Neither pool is accepted as confirmation data yet. Before choosing one, the
audit must search this project and the related AmanaMT work for any earlier use
of its candidate-level outcomes, verify machine-system and document coverage
using the outcome-free WMT systems XML, pin both repository revisions, and
record file hashes. WMT24 Japanese--Chinese is ineligible because it was used
in the completed sign-screen confirmation.

Public metadata sources:

- `https://github.com/wmt-conference/wmt24-news-systems/tree/main/xml`
- `https://github.com/google/wmt-mqm-human-evaluation/tree/main/generalMT2024`

## Claim boundary

This experiment evaluates pressure selection in a fixed candidate pool. It does
not establish a safeguard for generative reinforcement learning. Filtering
conditions describe dataset-slice exclusions and do not prove that individual
human annotations are incorrect.
