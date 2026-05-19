============================================================
ZENODO METADATA — TWO PAPER DEPOSITS
============================================================

This document contains complete metadata for two separate Zenodo
records, ready to copy-paste into the Zenodo upload form. Each
record is filled in independently. Items marked PLACEHOLDER must
be replaced before clicking Publish — they cannot be filled in
until the corresponding upload step is completed (see UPLOAD
SEQUENCE at the end).


============================================================
RECORD 1 — EXPERIMENTAL PAPER
============================================================

--- Upload step ---

Files: three_failure_modes_lstm2.pdf
DOI: Reserve DOI (click the "Reserve DOI" button on the upload
     form before filling in the rest — the reserved DOI must be
     inserted into the paper's title page and bibliography before
     final upload)

--- Bibliographic fields ---

Resource type: Publication
Publication type: Preprint
Title: Three Measurable Failure Modes of Large Language Models: Structure of the Error Distribution in Autoregressive Stochastic Models
Publication date: 2026-05-11
Language: English
Version: 1.0.0

--- Creators ---

Creator 1:
  Family name: Hubka
  Given names: Marek
  Name (display): Hubka, Marek
  Affiliation: Independent Researcher, Czech Republic
  ORCID: PLACEHOLDER — fill in author's ORCID (format: 0000-0000-0000-0000) before publishing

--- Description ---

Description (HTML allowed — paste as plain paragraphs):

This paper argues that "hallucination" in large language models is not a single phenomenon but three structurally distinct failure modes, each with a different cause, a different measurable signature, and a different class of effective intervention. Mode 1 (autoregressive reinforcement) is the self-consistent wrong trajectory produced when an error contaminates the model's own conditioning context. Mode 2 (confabulation) is fluent generation produced from parameter directions that received no training signal — the null space of the weight matrix. Mode 3 (irreducible uncertainty) is the correct response of a calibrated probabilistic system to a genuinely ambiguous query.

Each mode has a quantitative metric computable from API access: correction sensitivity (CS), dimensional excess (DE), and output entropy (H_out). The three measurements rest on a single coding-theoretic construction, the syndrome table S = N((J̄·V)ᵀ), whose full derivation is in the companion paper "A Syndrome Algebra for Differentiable Parametric Systems".

A controlled experimental series on a synthetic LSTM (D=256, L=10, six fixed seeds) confirms the framework end to end. The three metrics separate cleanly: the CS gap between known and unknown domains narrows monotonically from 0.273 ± 0.095 at k=1 to 0.067 ± 0.037 at k=10. The Pearson correlation r(DE, CS_unknown) = 0.9896 across k predicts out-of-domain failure from weights alone. Causal localisation of an injected perturbation reaches 100% accuracy over 180/180 trials with a pre/post residual ratio of approximately 2×10⁸. Oracle correction is exact (cosine 1.000000 over 36,000 trials). A direct comparison of multicellular specialists against monolithic generalists shows the Singleton-bound multicellular advantage grows from 0.158 ± 0.049 at N=5 to 0.310 ± 0.054 at N=10 in CS gap, empirically justifying the modular hierarchy.

Code and reproducibility scripts are available at https://github.com/PLACEHOLDER/gaia-lstm2.

--- Keywords ---

(Enter each on its own line in the Zenodo keywords field)

syndrome algebra
large language models
hallucination
error correction
Gram metric
Jacobian variance
modular architecture
Singleton bound
LSTM
machine learning reliability

--- License ---

License: Creative Commons Attribution 4.0 International (CC-BY-4.0)

--- Related identifiers ---

Related identifier 1:
  Relation: is supplement to
  Scheme: DOI
  Identifier: PLACEHOLDER — fill in the reserved DOI of the companion paper "A Syndrome Algebra for Differentiable Parametric Systems" (Record 2)
  Resource type: Publication / Preprint

Related identifier 2:
  Relation: is supplemented by
  Scheme: DOI
  Identifier: PLACEHOLDER — fill in the DOI of the code/data repository after the GitHub Release v1.0.0 has been created and Zenodo has minted the software DOI
  Resource type: Software

Related identifier 3:
  Relation: is supplemented by
  Scheme: URL
  Identifier: https://github.com/PLACEHOLDER/gaia-lstm2
  Resource type: Software
  Note: PLACEHOLDER — replace "PLACEHOLDER" in the URL with the actual GitHub organisation or user name before publishing

--- Communities ---

(Add each one in turn in the Communities field. Each requires
acceptance by the community curator after submission — submission
itself succeeds immediately.)

Community 1: zenodo
  Status: confirm availability (general community, normally open)

Community 2: machine-learning
  Status: confirm availability — search the Zenodo communities
  directory at https://zenodo.org/communities/ for the exact slug
  before submitting; if no exact match exists, omit this entry

Community 3: artificial-intelligence
  Status: confirm availability — same procedure as above

--- Funding ---

Funding: None (independent research)

--- Notes ---

Additional notes:

This preprint accompanies the companion mathematical paper "A Syndrome Algebra for Differentiable Parametric Systems" (see related identifiers). Code and data are available at the linked GitHub repository. Model weights are not included due to size; they are regenerated deterministically from the provided scripts and canonical seeds.


============================================================
RECORD 2 — COMPANION PAPER
============================================================

--- Upload step ---

Files: syndrome_algebra_companion.pdf
DOI: Reserve DOI (click the "Reserve DOI" button before filling in
     the rest — same procedure as Record 1)

--- Bibliographic fields ---

Resource type: Publication
Publication type: Preprint
Title: A Syndrome Algebra for Differentiable Parametric Systems
Publication date: 2026-05-11
Language: English
Version: 1.0.0

--- Creators ---

Creator 1:
  Family name: Hubka
  Given names: Marek
  Name (display): Hubka, Marek
  Affiliation: Independent Researcher, Czech Republic
  ORCID: PLACEHOLDER — fill in author's ORCID (format: 0000-0000-0000-0000) before publishing

--- Description ---

Description (HTML allowed — paste as plain paragraphs):

This is a mathematics paper. It constructs a syndrome algebra for differentiable parametric systems by generalising the syndrome-measurement principle of quantum error-correcting codes from a discrete binary setting to a continuous real-valued one. The result is a tool for diagnosing the response of any differentiable system to weight-space perturbations; its application to neural network reliability is the subject of the companion paper "Three Measurable Failure Modes of Large Language Models" and is not the topic here.

The starting point is the [[5,1,3]]₂ stabilizer code of Bennett et al., whose parity-check matrix maps single-qubit errors to four-bit syndromes. Three replacements take this construction to the continuous setting. (1) Field replacement: the finite-field arithmetic over F₂ is replaced by row-wise ℓ₂-normalisation over ℝ. (2) Basis replacement: the symplectic Pauli error basis is replaced by the right singular vectors of the weight matrix (the SVD basis). (3) Map replacement: the parity-check matrix is replaced by the averaged Jacobian of the logit map at probe inputs. Under these three substitutions, the syndrome table becomes S = N((J̄·V)ᵀ), and the discrete [[5,1,3]]₂ code is recovered as the q=2 special case.

Seven theorems establish the algebra's main properties. Theorem 1: the Jacobian is the syndrome table in the linear regime. Theorem 2: oracle correction is exact on the linear path. Theorem 3: crossing error (correction in a wrong direction) is positive in expectation. Theorem 4: the null space of the weight matrix is the confabulation subspace. Theorem 5: multi-layer measurement achieves identifiability. Theorem 6 (Jacobian Uncertainty Principle): the product of dictionary variance and observation variance is bounded below by a quantity proportional to the per-direction Jacobian variance. Theorem 7 (Reliability Uncertainty Principle): for any [[n,k,d]]_q code, normalised robustness, capacity, and expressiveness satisfy a structural trade-off that no architecture can escape.

The intended audience is mathematicians, physicists, and coding theorists. The paper is self-contained; no prior reading of the experimental companion is required.

--- Keywords ---

(Enter each on its own line in the Zenodo keywords field)

syndrome algebra
quantum error-correcting codes
[[5,1,3]] stabilizer code
Gram metric
Jacobian uncertainty principle
differentiable systems
error correction
coding theory
continuous limit
probabilistic systems

--- License ---

License: Creative Commons Attribution 4.0 International (CC-BY-4.0)

--- Related identifiers ---

Related identifier 1:
  Relation: is supplement to
  Scheme: DOI
  Identifier: PLACEHOLDER — fill in the reserved DOI of the experimental paper "Three Measurable Failure Modes of Large Language Models" (Record 1)
  Resource type: Publication / Preprint

Related identifier 2:
  Relation: is supplemented by
  Scheme: DOI
  Identifier: PLACEHOLDER — fill in the DOI of the code/data repository after the GitHub Release v1.0.0 has been created and Zenodo has minted the software DOI
  Resource type: Software

Related identifier 3:
  Relation: is referenced by
  Scheme: URL
  Identifier: https://github.com/PLACEHOLDER/gaia-lstm2
  Resource type: Software
  Note: PLACEHOLDER — replace "PLACEHOLDER" in the URL with the actual GitHub organisation or user name before publishing

--- Communities ---

(Same procedure as Record 1 — add each community in turn,
acceptance is curated.)

Community 1: zenodo
  Status: confirm availability (general community, normally open)

Community 2: mathematics
  Status: confirm availability — search Zenodo communities directory
  for the exact slug before submitting

Community 3: quantum-information
  Status: confirm availability — same procedure

Community 4: machine-learning
  Status: confirm availability — same procedure

--- Funding ---

Funding: None (independent research)

--- Notes ---

Additional notes:

This paper presents the mathematical foundation for the syndrome algebra framework. Experimental validation on neural networks is provided in the companion paper "Three Measurable Failure Modes of Large Language Models" (see related identifiers). The paper is self-contained: the complete derivation from the [[5,1,3]]₂ quantum error-correcting code to the continuous real-valued syndrome table is provided, with all proofs in the main body.


============================================================
UPLOAD SEQUENCE
============================================================

The upload has a chicken-and-egg structure: the papers cite each
other's DOIs and the code/data DOI, but no DOI exists until the
upload has happened. Zenodo solves this with the "Reserve DOI"
button, which mints the DOI without publishing the record. The
sequence below puts every step in the right order.

--- Step 1: Reserve both paper DOIs (no PDFs uploaded yet) ---

1.1  Log in to https://zenodo.org with the account that will own
     the papers.

1.2  Click "New upload". Do not upload the PDF yet. Click "Reserve
     DOI" near the top of the form. The form displays a reserved
     DOI of the form 10.5281/zenodo.XXXXXXXX.

1.3  Note the reserved DOI for Record 1 (experimental paper). Save
     the form as a draft (button at the bottom: "Save"). Do NOT
     click "Publish" — drafts are private and can be edited; once
     published, the file list is fixed and the record is public.

1.4  Open a second "New upload" tab and repeat for Record 2
     (companion paper). Note the second reserved DOI.

At the end of Step 1 you have two reserved DOIs, both drafts, no
PDFs uploaded.

--- Step 2: Insert reserved DOIs into the papers ---

2.1  Open three_failure_modes_lstm2.tex. In the bibliography,
     replace the bibitem placeholder for HubkaAlgebra with the
     reserved DOI of Record 2.

2.2  Open syndrome_algebra_companion.tex. Replace the analogous
     placeholder with the reserved DOI of Record 1.

2.3  Optionally insert "Zenodo DOI: 10.5281/zenodo.XXXXXXXX" on
     each paper's title page or in a footnote on the first page.

2.4  Recompile both PDFs. These are the final versions to upload.

--- Step 3: Upload PDFs and complete metadata for paper records ---

3.1  Return to the Record 1 draft on Zenodo. Drag the final PDF
     three_failure_modes_lstm2.pdf into the file upload area.
     Wait for the upload to complete and the preview to appear.

3.2  Fill in every field from the Record 1 block above, in order:
     Resource type → Publication type → Title → Publication date →
     Language → Version → Creators → Description → Keywords →
     License → Related identifiers → Communities → Funding →
     Notes.

3.3  For related identifiers, the code/data DOI is still PLACEHOLDER
     at this stage. Leave it as PLACEHOLDER and come back in Step 6
     to fill it in. Add the companion paper DOI (Record 2's reserved
     DOI) and the GitHub URL now.

3.4  Click "Save". Do NOT click "Publish" yet.

3.5  Repeat for Record 2.

At the end of Step 3 both drafts have PDFs and most metadata, with
the code/data DOI still missing.

--- Step 4: Update GitHub metadata files ---

4.1  In the GitHub repository at https://github.com/PLACEHOLDER/gaia-lstm2,
     update README.md with both reserved paper DOIs as Zenodo badges
     at the top of the file.

4.2  Update CITATION.cff with the same two DOIs. Set version to
     1.0.0 and date-released to the current date.

4.3  Commit and push.

--- Step 5: Create GitHub Release v1.0.0 ---

5.1  On GitHub, navigate to Releases → Draft a new release.

5.2  Tag: v1.0.0. Title: v1.0.0. Description: brief release notes
     plus the two paper DOIs.

5.3  Publish the release. If Zenodo-GitHub integration is enabled
     on this repository, Zenodo will automatically create a software
     record and mint a DOI for it within a few minutes. If the
     integration is not enabled, enable it now (Zenodo → Settings →
     GitHub → toggle the repository on, then re-publish the release
     by deleting and recreating it, or by clicking "Re-sync" in
     Zenodo).

5.4  Note the software DOI minted by Zenodo. This is the code/data
     DOI used in Step 6.

--- Step 6: Fill in the code/data DOI in both paper records ---

6.1  Return to Record 1 draft on Zenodo. Open the related
     identifiers section. Replace the PLACEHOLDER on the "is
     supplemented by" / Software DOI entry with the software DOI
     from Step 5.

6.2  Verify the GitHub URL placeholder is also replaced with the
     real repository URL.

6.3  Save the draft. Do NOT publish yet.

6.4  Repeat for Record 2.

--- Step 7: Final review and publish ---

7.1  Run the FINAL CHECKLIST below against each paper record.

7.2  When all checklist items pass for both records, click
     "Publish" on each.

7.3  The software record from Step 5 is already published (GitHub
     releases auto-publish on Zenodo). Edit it on Zenodo and add
     "is supplement to" related identifiers pointing to both paper
     DOIs, so the cross-linking is bidirectional.

7.4  arXiv submission (optional but recommended): with all three
     Zenodo records public and cross-linked, submit the two PDFs
     to arXiv. The Zenodo DOIs are already in the papers.

--- A note on drafts ---

Zenodo drafts can be saved indefinitely and edited freely. Once
published, the file list is immutable: a new file requires a new
version (which gets a new DOI suffix). The metadata of a published
record CAN be edited, so updating cross-link related identifiers
after publication is fine — but the PDFs themselves cannot be
replaced without a new version.


============================================================
FINAL CHECKLIST
============================================================

--- For each paper record, before clicking Publish ---

[ ] Title matches the paper exactly, including subtitle
[ ] Author name spelled correctly
[ ] ORCID filled in (not PLACEHOLDER)
[ ] Affiliation filled in
[ ] Resource type: Publication / Preprint (not Journal article)
[ ] License: CC BY 4.0
[ ] Description is 250–350 words, standalone
[ ] All 10 keywords entered, one per line
[ ] Related identifier 1: companion paper DOI (not PLACEHOLDER)
[ ] Related identifier 2: code/data DOI (not PLACEHOLDER)
[ ] Related identifier 3: GitHub URL with real org/user name
    (not "PLACEHOLDER" anywhere in the URL)
[ ] PDF uploaded and preview displays correctly
[ ] DOI reserved (shown on the form before publishing)
[ ] Draft saved at least once before publishing

--- Before publishing either paper record ---

[ ] Both PDFs are final versions (not drafts of the paper itself)
[ ] Both DOIs have been reserved on Zenodo
[ ] Both DOIs have been inserted into both papers' bibliographies
[ ] GitHub README.md updated with both paper DOIs
[ ] CITATION.cff updated with both paper DOIs
[ ] GitHub Release v1.0.0 created
[ ] Code/data DOI received from Zenodo (Step 5 of upload sequence)
[ ] Code/data DOI inserted as related identifier in both paper
    records before publishing

--- After publishing all three records ---

[ ] Both paper records publicly accessible at their Zenodo URLs
[ ] Code/data record publicly accessible
[ ] All three records cross-linked in both directions (each record's
    "Related identifiers" section lists the other two)
[ ] arXiv submission prepared, with the Zenodo DOIs already in the
    papers' bibliographies and title pages
