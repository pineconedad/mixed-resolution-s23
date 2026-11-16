# Storyboard: *Views or Pixels?* (CVPR 2027)

Working notes for drafting `paper2/`. Sources: `paper/notes/PAPER_BRIEF_2026-11-16.md` and `reports/STUDY_REPORT.md` in the paper-2 repository (`MadLad10/feedforward_3dcnn`), its readouts and protocols, and the existing draft on `claude/study-report-and-paper-fixes`. No number in the paper may come from anywhere else. Paper 1's numbers (this repository) use a different pipeline and are never mixed in.

## 1. The story in one paragraph

Feed-forward 3D models spend a token budget. Mixing a few sharp views with many small ones looks like a smart way to spend it, and mixed input also happens on its own. All three models we test collapse on it. We ask why, by changing one factor at a time on fixed frames. Upsampling the small views adds no information but brings π³ and DA3 back near the all-sharp result, so their failure is mostly about how the views are represented, not what they contain. DA3 has no positions in global attention and still collapses, so misaligned cross-view positions are not the whole story. Fixes inside the model recover little; resizing every view to one size within the mix's own cost recovers most. VGGT differs and we say so. Finally, when tokens can be spent freely, the advantage people expect from mixing is an advantage of having more views.

## 2. Structure, modelled on CVPR analysis papers

Lessons taken from recent CVPR papers. VGGT (CVPR 2025 best paper) states one simple question early, shows a teaser that answers it in one picture, and keeps the method section short. Question-driven analysis papers at CVPR state each finding as a section or paragraph heading and follow it with the single most direct piece of evidence. So:

| Section | Job | Main evidence | Figure / table |
| --- | --- | --- | --- |
| Teaser | Show the collapse and the upsampling recovery in one picture | π³ / DA3 / VGGT on HiRoom, ρ = 2 | Fig. 1 (`fig_teaser`) |
| 1 Intro | Why budgets, why mixing, what fails, what we did, three findings, contributions | S23 headline, S25, S24 | — |
| 2 Related | Three threads: feed-forward models; efficiency for them; resolution and positions in transformers | Reading notes | — |
| 3 Setup | Models, budget, mixed inputs, interventions, data, metrics, protocol, cost | Model table, timing (S8) | Fig. 2 (`fig_arch`), Tab. 1 |
| 4 Why they fail | F1 collapse, F2 representation, F3 DA3 control, F5 VGGT, robustness, replication | S23, S26, S27, S30, S31, S33 | Tab. 2, Tab. 3, Fig. 3, Fig. 4 |
| 5 What fixes it | Positions, attention, one size at equal cost, equal detail, shrinking, how many views, recipe | S5, S10–S17, S25, S28, S29, S32 | Fig. 5, Tab. 4 |
| 6 Views or pixels | Allocation: frozen U vs X, then control view count | S7, S20–S22, S24 | Fig. 6, Tab. 5 |
| 7 Discussion | What it means, failed predictions, limitations | All | — |

Figure `fig_qual` goes to the supplementary material for space.

## 3. Claims and the exact wording allowed

| Claim | Allowed wording | Not allowed |
| --- | --- | --- |
| Collapse | "All three models collapse" with the HiRoom example | "Models cannot use mixed input" |
| Small views hurt sharp ones | "For pose, scored on the four shared sharp frames, the aligned mix is below the four sharp views alone in 28 of 30 cells (17 significant, 1 significant the other way)" | "Small views always hurt" (reconstruction is less one-sided) |
| Representation | "For π³ and DA3, mostly representation" | Any unqualified "the failure is representational" |
| Recovery | "recovers 62–110% of the pose loss" | "fully fixes" |
| DA3 control | "cross-view position misalignment is not the whole cause" | "positions do not matter" |
| VGGT | "we do not establish why VGGT differs" | Any causal claim about the reference view |
| Shrinking | "depends on the absolute size of the small views; never beats the four sharp views alone (0 W / 53 L of 60)" | "the mix is worse than shrinking everything" |
| Allocation | "a uniform input with the right number of views is the better default" | "mixing is useless" (VGGT ScanNet++ 8k is an exception) |
| Counts | Always "W wins and L losses of N" | Wins without losses |

## 4. How numbers are presented

- Each number appears once where it carries the argument, with what it measures in the same sentence.
- Win/loss counts are explained the first time ("significant at two standard errors of a paired scene bootstrap; the rest inconclusive") and then used sparingly.
- Ranges are given for effect sizes; single examples use the HiRoom ρ = 2 scene set because the teaser uses it.
- Tables carry the bulk of the counts so the text can stay readable.

## 5. Facts checked in the code or data while drafting

- Upsampling in the code is **bicubic** (`src/mixres/pose_jobs.py`, D4 and S26 arms). The older draft said bilinear; the new draft says bicubic.
- The 16,384-token allocation run (S18) used the selection seed only, so it is reported as such.
- Five bibliography entries had placeholder author lists. Filled from the reading notes and arXiv: ReDT, HD-VGGT, Good Token Hunting, CRPA. *Many Eyes, One World* could not be verified and is not cited.

## 6. Open items the authors must still do (outside the text)

- Compile and check the 8-page limit; no LaTeX toolchain was available while drafting.
- Confirm the full title of Good Token Hunting (arXiv 2605.23892) in the bibliography.
- Write the supplementary material the paper points to: per-cell tables, protocols, timing table, replication table, shrink-everything table, qualitative figure.
