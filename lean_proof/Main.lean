import RewriteCollapse.lemmas
import RewriteCollapse.theorems

/-- Master theorem: going to the collapseical result never increases `E_down`. -/
theorem generalcollapseical {α : Type _} [DecidableEq α]
  (R : RuleSet α) (μ : α → Nat) (S : Finset α) :
  E_down (collapseResult μ S R) μ S ≤ E_down R μ S ∧ True :=
    ⟨collapseMinimal μ S R, trivial⟩
