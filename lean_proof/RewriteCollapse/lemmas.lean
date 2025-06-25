import Mathlib

namespace RewriteCollapse

/-- One rewrite rule `lhs → rhs`. -/
structure Rule (α : Type) where
  lhs : α
  rhs : α
  deriving DecidableEq

@[inline]
def Rule.apply {α : Type} [DecidableEq α] (r : Rule α) (x : α) : α :=
  if x = r.lhs then r.rhs else x

/-- A finite set of rewrite rules. -/
structure RuleSet (α : Type) where
  rules : List (Rule α)
  deriving DecidableEq

/-- Apply every rule once, left to right. -/
def applyOnce {α : Type} [DecidableEq α] (R : RuleSet α) (x : α) : α :=
  R.rules.foldl (fun a r => r.apply a) x

/-- A single "root" rewrite step. -/
def rewriteStep {α : Type} [DecidableEq α] (R : RuleSet α) (x y : α) : Prop :=
  ∃ r ∈ R.rules, x = r.lhs ∧ y = r.rhs

/-- Try to take one rewrite step from `x`. -/
def applyStep? {α : Type} [DecidableEq α] (R : RuleSet α) (x : α) : Option α :=
  let y := R.rules.foldl (fun a r => if a = r.lhs then r.rhs else a) x
  if y ≠ x then some y else none

/-- Repeatedly apply `applyStep?` up to `k` times. -/
def applyBounded {α : Type} [DecidableEq α] (R : RuleSet α) : Nat → α → α
| 0,     x => x
| k + 1, x =>
  match applyStep? R x with
  | none   => x
  | some y => applyBounded R k y

/-- The full collapse map: apply at most `μ x + 1` steps. -/
noncomputable def collapseMap {α : Type} [DecidableEq α]
  (R : RuleSet α) (μ : α → Nat) : α → α
| x => applyBounded R (μ x + 1) x

open BigOperators

/-- "Downwards" collapse metric: sum of `log (|fiber| + 1)`. -/
noncomputable def E_down {α : Type} [DecidableEq α]
  (R : RuleSet α) (μ : α → Nat) (S : Finset α) : Real :=
  let imgs := S.image (collapseMap R μ)
  ∑ c ∈ imgs, Real.log (↑(S.filter fun x => collapseMap R μ x = c).card + 1)

/-- How many rules are in the set? -/
def ruleCount {α : Type} (R : RuleSet α) : Nat :=
  R.rules.length

/-- Simple helper: empty rule set has zero rules. -/
lemma empty_ruleCount {α : Type} : ruleCount (⟨[]⟩ : RuleSet α) = 0 := by
  simp [ruleCount]

lemma single_ruleCount {α : Type} (r : Rule α) :
  ruleCount ⟨[r]⟩ = 1 := by simp [ruleCount]

/-- Try to delete `r` if it does not increase `E_down`. -/
noncomputable def collapseStep {α : Type} [DecidableEq α]
  (μ : α → Nat) (S : Finset α) (R : RuleSet α) (r : Rule α) : RuleSet α :=
  if E_down { rules := R.rules.erase r } μ S ≤ E_down R μ S then
    { rules := R.rules.erase r }
  else
    R

/-- One left-to-right sweep over all rules. -/
noncomputable def collapsePass {α : Type} [DecidableEq α]
  (μ : α → Nat) (S : Finset α) (R : RuleSet α) : RuleSet α :=
  R.rules.foldl (fun acc r => collapseStep μ S acc r) R

/-- Iterate `collapsePass` to a fix-point. -/
noncomputable def collapseFix {α : Type} [DecidableEq α]
  (μ : α → Nat) (S : Finset α) : RuleSet α → RuleSet α :=
  collapsePass μ S

end RewriteCollapse
