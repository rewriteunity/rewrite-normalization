import RewriteCollapse.lemmas

namespace RewriteCollapse

/-- Deleting a rule never increases `E_down`. -/
theorem collapseStep_le {α : Type _} [DecidableEq α]
  (μ : α → Nat) (S : Finset α)
  (R : RuleSet α) (r : Rule α) :
  E_down (collapseStep μ S R r) μ S ≤ E_down R μ S := by
  dsimp [collapseStep]
  by_cases h : E_down { rules := R.rules.erase r } μ S ≤ E_down R μ S
  · -- then: we picked the erased set, so `h` is exactly the needed inequality
    simp [h]
  · -- else: we kept `R` unchanged, so the metric is definitionally the same
    simp [h]

/-- Folding a monotone step function does not increase the metric. -/
theorem foldl_monotone {α : Type _} [DecidableEq α]
  (μ : α → Nat) (S : Finset α)
  (f : RuleSet α → Rule α → RuleSet α)
  (H : ∀ R r, E_down (f R r) μ S ≤ E_down R μ S)
  (R : RuleSet α) (l : List (Rule α)) :
  E_down (l.foldl f R) μ S ≤ E_down R μ S := by
  induction l generalizing R with
  | nil =>
      -- base case: `foldl` on `[]` returns `R`
      simp
  | cons r rs ih =>
      -- inductive step: do one `f`‐step, then recurse
      simp [List.foldl]
      apply le_trans (ih (f R r)) (H R r)

/-- The collapse result after full elimination. -/
noncomputable abbrev collapseResult {α : Type} [DecidableEq α]
  (μ : α → Nat) (S : Finset α) (R : RuleSet α) : RuleSet α :=
  collapseFix μ S R

/-- The collapse metric never increases under collapse. -/
theorem collapseMinimal {α : Type _} [DecidableEq α]
  (μ : α → Nat) (S : Finset α) (R : RuleSet α) :
  E_down (collapseResult μ S R) μ S ≤ E_down R μ S := by
  dsimp only [collapseResult, collapseFix, collapsePass]
  exact foldl_monotone μ S (collapseStep μ S) (collapseStep_le μ S) R R.rules



end RewriteCollapse
