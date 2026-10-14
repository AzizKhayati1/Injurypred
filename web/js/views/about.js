import { getMeta } from "../data.js";
import { esc, fmtDate } from "../ui.js";

const REPO = "https://github.com/AzizKhayati1/Injurypred";

export async function aboutView(ctx) {
  const meta = await getMeta();
  ctx.title("Methodology");
  ctx.crumbs([["Model"], ["Methodology"]]);
  const lt = meta.evaluation.latent_trait_recovery;
  ctx.main.innerHTML = `<article class="prose">
    <h1>Methodology</h1>
    <p class="summary" style="margin-top:14px"><strong>This project estimates statistical availability risk from workload patterns. It is not a medical diagnostic system.</strong>
      All athletes and data are synthetic.</p>
    <h2>What is estimated</h2>
    <p>For each athlete who is available at the end of day <em>t</em>: the probability that a new time-loss absence (injury or illness) begins in
      the following 7 days, using only data dated on or before <em>t</em>. Probabilities are calibrated; categories are multiples of the reference base rate
      (Moderate ≥ 1.5×, High ≥ 2.5×, Very high ≥ 4×).</p>
    <h2>Workload features</h2>
    <p>76 features from trailing windows: 7/14/28-day loads (distance, high-speed running, sprints, accelerations, player load, session-RPE), several
      acute:chronic ratios (coupled, uncoupled, EWMA), week-over-week change, monotony and strain, recovery (rest days, consecutive sessions),
      injury history, and <strong>athlete-specific baselines</strong>: each athlete's normal week over the previous 12 weeks (ending 7 days ago,
      absence days excluded), with % and z-score deviations.</p>
    <h2>The ACWR debate</h2>
    <p>The acute:chronic workload ratio is widely used but contested: the conventional version is mathematically coupled (Lolli et al., 2019),
      ratios are unstable when chronic load is low, "sweet-spot" zones are fragile, and observational associations are not causal
      (Impellizzeri et al., 2020; 2021). Here ACWR is one candidate among many; a head-to-head experiment found that no representation clearly
      dominates once alternatives are available.</p>
    <h2>Validation without leakage</h2>
    <ul>
      <li>Chronological splits: train on season 2024-25, validate on Jul–Dec 2025, test on Jan–May 2026; the current season is monitored.</li>
      <li>Purging: a row is used only if its 7-day label window ends inside its own split.</li>
      <li>A test recomputes features from data truncated at <em>t</em> and checks they are identical — no future information.</li>
      <li>Forward (month-by-month) validation, an unseen-athlete hold-out and athlete-level bootstrap confidence intervals.</li>
    </ul>
    <h2>Explanations</h2>
    <p>SHAP contributions are shown as factors <em>associated with</em> higher or lower estimated risk, compared with the athlete's own history.
      They describe the model, not causes; the what-if simulator likewise shows how the model responds, not what would happen to the athlete.</p>
    <h2>Limitations</h2>
    <ul>
      <li>Discrimination is modest (test ROC-AUC ≈ 0.65): the model helps prioritise attention, not decide about individuals.</li>
      ${lt ? `<li>On the synthetic data, an athlete's average estimate tracks their hidden injury susceptibility (ρ = ${lt.spearman_injury_susceptibility.toFixed(2)})
        but not their hidden workload tolerance (ρ = ${lt.spearman_workload_tolerance.toFixed(2)}).</li>` : ""}
      <li>Contact injuries and most illnesses are not load-driven; the model does not know who will play next week.</li>
      <li>Real deployments involve health data: consent, purpose limitation, access control and human review are required.</li>
    </ul>
    <h2>This site</h2>
    <p>This is a static build: every number was pre-computed by the Python pipeline (data ${esc(meta.model.data_range?.join(" – ") || "")}, exported
      ${fmtDate(meta.generated_at?.slice(0, 10))}). The full system — FastAPI service, Streamlit dashboard, MLflow, PostgreSQL and Docker Compose — is in the
      <a href="${REPO}" target="_blank" rel="noopener">repository</a>; see <code>README.md</code>, <code>docs/methodology.md</code> and <code>docs/model_card.md</code>.</p>
  </article>`;
}
