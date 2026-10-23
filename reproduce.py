# %% [markdown]
# # Developer Experience vs. Bug-Inducing Rate in AI-Assisted PRs
#
# Secondary analysis on the RQ3 dataset from Mazloomzadeh et al. ([arXiv:2607.21832](https://arxiv.org/abs/2607.21832), [GitHub repo](https://github.com/mazloomzadeh/AI-Agent-Analysis)).
#
# Downloaded files and the merged cohort table are cached in `data_cache/`. Delete a file there if you want to re-download or rebuild it.

# %%
from pathlib import Path
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)

AGENTS = ["claude", "codex", "cursor"]
PR_URL_RE = r"(?:repos|github\.com)/([^/]+/[^/]+)/pulls?/(\d+)"


def get_csv(url, filename):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Downloading {filename}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            path.write_bytes(resp.read())
    return path


# %% [markdown]
# ## 1. Download dataset files
#
# We exclude `copilot` and `devin` because their PRs are authored by bot accounts (`Copilot` and `devin-ai-integration[bot]`), so we can't track the underlying developer's prior PR history.

# %%
BASE_RAW = "https://raw.githubusercontent.com/mazloomzadeh/AI-Agent-Analysis/main"
BASE_LFS = "https://media.githubusercontent.com/media/mazloomzadeh/AI-Agent-Analysis/main"

bug_csv = get_csv(
    f"{BASE_RAW}/RQ3/Analysis_Part2/which_prs_are_bug_inducing_which_not_all_agents.csv",
    "bug_inducing.csv",
)

info_csvs = {
    a: get_csv(
        f"{BASE_RAW}/RQ3/Analysis_Part1/prs_general_info_with_contributors/{a}_prs_general_info_with_comments_contributors_count.csv",
        f"{a}_info.csv",
    )
    for a in AGENTS
}

all_prs_csvs = {
    a: get_csv(
        f"{BASE_LFS}/all_prs/{a}_filtered_prs_info_between_release_date_and_cutoff_date.csv",
        f"{a}_all_prs.csv",
    )
    for a in AGENTS
}

# %% [markdown]
# ## 2. Build cohort and point-in-time experience metrics
#
# For each PR in the RQ3 sample ($N = 2,153$), we use the full repo PR history (`all_prs`) to compute two experience metrics strictly before that PR was created:
# - `log_prior_merged_prs`: $\ln(\text{prior merged PRs by this author in this repo} + 1)$
# - `log_days_first_merged`: $\ln(\text{days since author's first merged PR} + 1)$

# %%
cohort_csv = DATA_DIR / "rq3_cohort.csv"

if not cohort_csv.exists():
    bug_df = pd.read_csv(bug_csv, usecols=["pr_url", "pr_type", "agent", "is_bug_inducing"])
    info_df = pd.concat(
        [
            pd.read_csv(
                info_csvs[a],
                usecols=["pr_url", "pr_type", "agent", "pr_username", "pr_usertype", "created_at"],
            )
            for a in AGENTS
        ],
        ignore_index=True,
    )

    rq3 = pd.merge(bug_df, info_df, on=["pr_url", "pr_type", "agent"])
    rq3 = rq3[rq3["agent"].isin(AGENTS) & (rq3["pr_usertype"] == "User")].copy()
    rq3["is_ai"] = (rq3["pr_type"] == "AM").astype(int)

    hist_cols = ["pr_url", "repo_fullname", "pr_user", "pr_user_type", "pr_created_date", "pr_merge_user"]
    all_prs = pd.concat([pd.read_csv(all_prs_csvs[a], usecols=hist_cols) for a in AGENTS], ignore_index=True)

    ext = all_prs["pr_url"].str.extract(PR_URL_RE)
    all_prs["pr_key"] = ext[0].str.lower() + "#" + ext[1]
    all_prs["repo_lower"] = all_prs["repo_fullname"].astype(str).str.lower()
    all_prs["user_lower"] = all_prs["pr_user"].astype(str).str.lower()
    all_prs["pr_created_dt"] = pd.to_datetime(all_prs["pr_created_date"], utc=True)
    all_prs["is_merged"] = all_prs["pr_merge_user"].notna().astype(int)

    unique_prs = (
        all_prs.drop_duplicates(subset=["pr_key"])
        .sort_values(["pr_created_dt", "pr_key"])
        .reset_index(drop=True)
    )

    # Prior merged PRs in the same repo
    unique_prs["prior_merged_prs"] = (
        unique_prs.groupby(["repo_lower", "user_lower"])["is_merged"].cumsum()
        - unique_prs["is_merged"]
    )
    unique_prs["log_prior_merged_prs"] = np.log(unique_prs["prior_merged_prs"] + 1)

    # Days since author's first merged PR
    first_merged = unique_prs[unique_prs["is_merged"] == 1].groupby("user_lower")["pr_created_dt"].min()
    days_elapsed = (unique_prs["pr_created_dt"] - unique_prs["user_lower"].map(first_merged)).dt.total_seconds() / 86400
    unique_prs["days_since_first_merged"] = np.maximum(0.0, days_elapsed.fillna(0.0))
    unique_prs["log_days_first_merged"] = np.log(unique_prs["days_since_first_merged"] + 1)

    rq3_ext = rq3["pr_url"].str.extract(PR_URL_RE)
    rq3["pr_key"] = rq3_ext[0].str.lower() + "#" + rq3_ext[1]
    rq3 = rq3.merge(
        unique_prs[["pr_key", "log_prior_merged_prs", "log_days_first_merged"]],
        on="pr_key",
        how="left",
    )
    rq3["is_bug"] = np.where(rq3["is_bug_inducing"] == "BI", 100.0, 0.0)
    rq3.to_csv(cohort_csv, index=False)

rq3 = pd.read_csv(cohort_csv)
rq3["repository"] = rq3["pr_url"].str.extract(PR_URL_RE)[0].str.lower()
if rq3["repository"].isna().any():
    raise ValueError("Could not extract a repository from every PR URL")

print(
    f"N = {len(rq3)} PRs "
    f"(Human: {(rq3['is_ai'] == 0).sum()}, AI: {(rq3['is_ai'] == 1).sum()}); "
    f"{rq3['repository'].nunique()} repository clusters"
)

# %% [markdown]
# ## 3. OLS interaction regressions
#
# Model: $\text{is\_bug} = \beta_0 + \beta_1\,\text{is\_ai} + \beta_2\,\text{Exp} + \beta_3\,(\text{is\_ai} \times \text{Exp})$
# - Human slope: $\beta_2$
# - AI slope: $\beta_2 + \beta_3$
# - Interaction (AI $-$ Human): $\beta_3$
#
# We report conventional p-values and p-values based on repository-clustered standard errors. Clustering allows observations within the same repository to have correlated errors.

# %%
metrics = {
    "log_prior_merged_prs": "Prior Merged PRs (ln(PRs + 1))",
    "log_days_first_merged": "Active Tenure (ln(Days + 1))",
}

axis_labels = {
    "log_prior_merged_prs": "Prior merged PRs",
    "log_days_first_merged": "Active tenure (days)",
}

rows = []
models = {}
clustered_models = {}
for col, label in metrics.items():
    m = smf.ols(f"is_bug ~ is_ai * {col}", data=rq3).fit()
    m_clustered = smf.ols(f"is_bug ~ is_ai * {col}", data=rq3).fit(
        cov_type="cluster", cov_kwds={"groups": rq3["repository"]}
    )
    models[col] = m
    clustered_models[col] = m_clustered
    t_ai = m.t_test(f"{col} + is_ai:{col} = 0")
    t_ai_clustered = m_clustered.t_test(f"{col} + is_ai:{col} = 0")
    inter = f"is_ai:{col}"

    rows.append(
        {
            "Metric": label,
            "Human Slope": (
                f"{m.params[col]:+.3f} "
                f"(p={m.pvalues[col]:.4f}; clustered p={m_clustered.pvalues[col]:.4f})"
            ),
            "AI Slope": (
                f"{t_ai.effect.item():+.3f} "
                f"(p={t_ai.pvalue.item():.4f}; clustered p={t_ai_clustered.pvalue.item():.4f})"
            ),
            "Interaction": (
                f"{m.params[inter]:+.3f} "
                f"(p={m.pvalues[inter]:.4f}; clustered p={m_clustered.pvalues[inter]:.4f})"
            ),
        }
    )

print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ## 4. Visualize experience slopes with and without AI
#
# Each graph shows the fitted bug-inducing rate from the corresponding interaction model. Because the models are linear in $\ln(\mathrm{experience} + 1)$, the horizontal axes use logarithmic spacing. Tick labels are converted back to actual PR counts and days.

# %%
colors = {"Without AI": "#4C78A8", "With AI": "#E45756"}

for col, label in metrics.items():
    log_x = np.linspace(rq3[col].min(), rq3[col].max(), 200)
    fig, ax = plt.subplots(figsize=(8, 5))

    for is_ai, group_label in [(0, "Without AI"), (1, "With AI")]:
        predictions = models[col].predict(pd.DataFrame({col: log_x, "is_ai": is_ai}))
        ax.plot(log_x, predictions, linewidth=2.5, color=colors[group_label], label=group_label)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _position: f"{np.expm1(value):,.0f}")
    )

    ax.set(
        title=f"Experience and Predicted Bug-Inducing Rate\n{axis_labels[col]}",
        xlabel=axis_labels[col],
        ylabel="Predicted bug-inducing rate (%)",
    )
    ax.legend(title="PR type", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Logistic interaction regressions
#
# Logistic regression models the probability that a PR is bug-inducing. The specification is the same as above:
#
# $\operatorname{logit}(P(\text{bug})) = \beta_0 + \beta_1\,\text{is\_ai} + \beta_2\,\text{Exp} + \beta_3\,(\text{is\_ai} \times \text{Exp})$
#
# Odds ratios below describe the multiplicative change in bug-inducing odds for a one-unit increase in $\ln(\mathrm{experience} + 1)$. The interaction odds ratio tests whether that experience effect differs between PRs with and without AI. McFadden's pseudo-$R^2$ summarizes overall model fit relative to an intercept-only model.
#
# The 95% confidence intervals and first p-values use conventional standard errors. The second p-values use repository-clustered standard errors.

# %%
rq3["is_bug_binary"] = (rq3["is_bug"] > 0).astype(int)

logit_rows = []
logit_models = {}
clustered_logit_models = {}

for col, label in metrics.items():
    m = smf.logit(f"is_bug_binary ~ is_ai * {col}", data=rq3).fit(disp=False)
    m_clustered = smf.logit(f"is_bug_binary ~ is_ai * {col}", data=rq3).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": rq3["repository"]},
    )
    logit_models[col] = m
    clustered_logit_models[col] = m_clustered

    ai_effect = m.t_test(f"{col} + is_ai:{col} = 0")
    ai_effect_clustered = m_clustered.t_test(f"{col} + is_ai:{col} = 0")
    ai_log_odds = ai_effect.effect.item()
    interaction = f"is_ai:{col}"
    without_ai_ci = np.exp(m.conf_int().loc[col].to_numpy())
    with_ai_ci = np.exp(np.asarray(ai_effect.conf_int()).ravel())
    interaction_ci = np.exp(m.conf_int().loc[interaction].to_numpy())

    logit_rows.append(
        {
            "Metric": label,
            "Without AI OR [95% CI]": (
                f"{np.exp(m.params[col]):.3f} "
                f"[{without_ai_ci[0]:.3f}, {without_ai_ci[1]:.3f}] "
                f"(p={m.pvalues[col]:.4f}; clustered p={m_clustered.pvalues[col]:.4f})"
            ),
            "With AI OR [95% CI]": (
                f"{np.exp(ai_log_odds):.3f} "
                f"[{with_ai_ci[0]:.3f}, {with_ai_ci[1]:.3f}] "
                f"(p={ai_effect.pvalue.item():.4f}; "
                f"clustered p={ai_effect_clustered.pvalue.item():.4f})"
            ),
            "Interaction OR [95% CI]": (
                f"{np.exp(m.params[interaction]):.3f} "
                f"[{interaction_ci[0]:.3f}, {interaction_ci[1]:.3f}] "
                f"(p={m.pvalues[interaction]:.4f}; "
                f"clustered p={m_clustered.pvalues[interaction]:.4f})"
            ),
            "McFadden R²": f"{m.prsquared:.4f}",
        }
    )

print(pd.DataFrame(logit_rows).to_string(index=False))

# %% [markdown]
# **Interpretation.** With conventional standard errors, greater prior-PR experience and longer active tenure significantly predict lower bug-inducing odds for PRs with AI. After clustering by repository, neither with-AI experience effect remains below the 0.05 threshold (prior PRs: $p = 0.1923$; active tenure: $p = 0.0576$). The active-tenure interaction remains significant with clustered standard errors ($p = 0.0187$), while the prior-PR interaction does not ($p = 0.1335$). Neither experience measure significantly predicts bug-inducing odds for PRs without AI under either inference method. The low McFadden pseudo-$R^2$ values (0.0100 and 0.0136) indicate that these small models have limited overall predictive power.

# %% [markdown]
# ## 6. Visualize logistic-regression predictions
#
# The curves show each model's predicted probability that a PR is bug-inducing. Shaded regions are 95% confidence intervals for the mean prediction based on repository-clustered standard errors. Horizontal-axis spacing follows $\ln(\mathrm{experience} + 1)$, while the tick labels show actual PR counts and days.

# %%
for col in metrics:
    model = logit_models[col]
    clustered_model = clustered_logit_models[col]
    log_x = np.linspace(rq3[col].min(), rq3[col].max(), 200)
    fig, ax = plt.subplots(figsize=(8, 5))

    for is_ai, group_label in [(0, "Without AI"), (1, "With AI")]:
        prediction_data = pd.DataFrame({col: log_x, "is_ai": is_ai})
        prediction = clustered_model.get_prediction(prediction_data).summary_frame()
        if is_ai == 0:
            effect_pvalue = model.pvalues[col]
            clustered_effect_pvalue = clustered_model.pvalues[col]
        else:
            effect_pvalue = model.t_test(f"{col} + is_ai:{col} = 0").pvalue.item()
            clustered_effect_pvalue = clustered_model.t_test(
                f"{col} + is_ai:{col} = 0"
            ).pvalue.item()

        ax.plot(
            log_x,
            100 * prediction["predicted"],
            linewidth=2.5,
            color=colors[group_label],
            label=(
                f"{group_label} (p={effect_pvalue:.3f}; "
                f"clustered p={clustered_effect_pvalue:.3f})"
            ),
        )
        ax.fill_between(
            log_x,
            100 * prediction["ci_lower"],
            100 * prediction["ci_upper"],
            color=colors[group_label],
            alpha=0.15,
        )

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _position: f"{np.expm1(value):,.0f}")
    )
    interaction = f"is_ai:{col}"
    ax.set(
        title=(
            f"Predicted Bug-Inducing Probability\n{axis_labels[col]}\n"
            f"AI × experience interaction: p={model.pvalues[interaction]:.3f}; "
            f"clustered p={clustered_model.pvalues[interaction]:.3f}"
        ),
        xlabel=axis_labels[col],
        ylabel="Predicted probability (%)",
    )
    ax.set_ylim(bottom=0)
    ax.legend(title="PR type", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Visualize the interaction contrast
#
# These contrast plots show the model-predicted difference in bug-inducing probability between otherwise equivalent PRs with and without AI. A changing difference across experience indicates an interaction on the probability scale. The shaded regions are approximate 95% confidence intervals based on repository-clustered standard errors; the horizontal line marks no predicted AI/non-AI difference.

# %%
for col in metrics:
    model = logit_models[col]
    clustered_model = clustered_logit_models[col]
    log_x = np.linspace(rq3[col].min(), rq3[col].max(), 200)
    interaction = f"is_ai:{col}"

    without_ai_data = pd.DataFrame({col: log_x, "is_ai": 0})
    with_ai_data = pd.DataFrame({col: log_x, "is_ai": 1})
    without_ai_probability = model.predict(without_ai_data)
    with_ai_probability = model.predict(with_ai_data)
    probability_difference = with_ai_probability - without_ai_probability

    without_ai_design = pd.DataFrame(
        {"Intercept": 1.0, "is_ai": 0.0, col: log_x, interaction: 0.0}
    )[model.params.index].to_numpy()
    with_ai_design = pd.DataFrame(
        {"Intercept": 1.0, "is_ai": 1.0, col: log_x, interaction: log_x}
    )[model.params.index].to_numpy()
    gradient = (
        with_ai_probability.to_numpy()[:, None]
        * (1 - with_ai_probability.to_numpy()[:, None])
        * with_ai_design
        - without_ai_probability.to_numpy()[:, None]
        * (1 - without_ai_probability.to_numpy()[:, None])
        * without_ai_design
    )
    covariance = clustered_model.cov_params().to_numpy()
    standard_error = np.sqrt(np.einsum("ij,jk,ik->i", gradient, covariance, gradient))
    lower = probability_difference - 1.96 * standard_error
    upper = probability_difference + 1.96 * standard_error

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(log_x, 100 * probability_difference, color="#725AC1", linewidth=2.5)
    ax.fill_between(log_x, 100 * lower, 100 * upper, color="#725AC1", alpha=0.18)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _position: f"{np.expm1(value):,.0f}")
    )
    ax.set(
        title=(
            f"Predicted AI − Non-AI Probability Difference\n{axis_labels[col]}\n"
            f"Log-odds interaction: p={model.pvalues[interaction]:.3f}; "
            f"clustered p={clustered_model.pvalues[interaction]:.3f}"
        ),
        xlabel=axis_labels[col],
        ylabel="Probability difference (percentage points)",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.show()
