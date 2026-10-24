# %% [markdown]
# # Developer Experience vs. Bug-Inducing Rate in AI-Assisted PRs
#
# Secondary analysis on the RQ3 dataset from Mazloomzadeh et al. ([arXiv:2607.21832](https://arxiv.org/abs/2607.21832), [GitHub repo](https://github.com/mazloomzadeh/AI-Agent-Analysis)).
#
# Downloaded files and the merged cohort table are cached in `data_cache/`. Delete a file there if you want to re-download or rebuild it.

# %%
from bisect import bisect_left
from pathlib import Path
import urllib.request
import matplotlib.pyplot as plt
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


def count_prior_merged_prs(pr_history):
    merged_dates = (
        pr_history.loc[pr_history["is_merged"].eq(1)]
        .groupby(["repo_lower", "user_lower"])["pr_closed_dt"]
        .apply(lambda dates: sorted(dates.dropna().tolist()))
        .to_dict()
    )

    # Count only merge timestamps strictly before each PR's creation time.
    # This excludes the current PR and earlier PRs that had not yet merged.
    return pd.Series(
        [
            bisect_left(merged_dates.get((repo, user), ()), created)
            for repo, user, created in zip(
                pr_history["repo_lower"],
                pr_history["user_lower"],
                pr_history["pr_created_dt"],
            )
        ],
        index=pr_history.index,
        dtype=int,
    )


# %% [markdown]
# ## Download dataset files
#
# We exclude `copilot` and `devin` because their PRs are generally authored by bot accounts (`Copilot` and `devin-ai-integration[bot]`), so we can't track the underlying developer's prior PR history. We retain the complete balanced RQ3 sample for Claude, Codex, and Cursor, including one Claude-labeled PR authored by the `Copilot` bot account.

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
# ## Build cohort and point-in-time experience
#
# For each PR in the selected RQ3 sample ($N = 2,154$), we use the full repo PR history (`all_prs`) to count the author's merged PRs in that repository strictly before the selected PR was created.

# %%
cohort_csv = DATA_DIR / "rq3_cohort_point_in_time.csv"

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
    rq3 = rq3[rq3["agent"].isin(AGENTS)].copy()
    rq3["is_ai"] = (rq3["pr_type"] == "AM").astype(int)

    hist_cols = [
        "pr_url",
        "repo_fullname",
        "pr_user",
        "pr_user_type",
        "pr_created_date",
        "pr_closed_date",
        "pr_merge_user",
    ]
    all_prs = pd.concat([pd.read_csv(all_prs_csvs[a], usecols=hist_cols) for a in AGENTS], ignore_index=True)

    ext = all_prs["pr_url"].str.extract(PR_URL_RE)
    all_prs["pr_key"] = ext[0].str.lower() + "#" + ext[1]
    all_prs["repo_lower"] = all_prs["repo_fullname"].astype(str).str.lower()
    all_prs["user_lower"] = all_prs["pr_user"].astype(str).str.lower()
    all_prs["pr_created_dt"] = pd.to_datetime(all_prs["pr_created_date"], utc=True)
    all_prs["pr_closed_dt"] = pd.to_datetime(all_prs["pr_closed_date"], utc=True)
    all_prs["is_merged"] = all_prs["pr_merge_user"].notna().astype(int)

    unique_prs = (
        all_prs.drop_duplicates(subset=["pr_key"])
        .sort_values(["pr_created_dt", "pr_key"])
        .reset_index(drop=True)
    )

    # Merges completed before this PR was created, excluding the current PR.
    unique_prs["prior_merged_prs"] = count_prior_merged_prs(unique_prs)
    unique_prs["log_prior_merged_prs"] = np.log(unique_prs["prior_merged_prs"] + 1)

    rq3_ext = rq3["pr_url"].str.extract(PR_URL_RE)
    rq3["pr_key"] = rq3_ext[0].str.lower() + "#" + rq3_ext[1]
    rq3 = rq3.merge(
        unique_prs[["pr_key", "prior_merged_prs", "log_prior_merged_prs"]],
        on="pr_key",
        how="left",
    )
    rq3["is_bug"] = np.where(rq3["is_bug_inducing"] == "BI", 100.0, 0.0)
    rq3.to_csv(cohort_csv, index=False)

rq3 = pd.read_csv(cohort_csv)
rq3["repository"] = rq3["pr_url"].str.extract(PR_URL_RE)[0].str.lower()
if rq3["repository"].isna().any():
    raise ValueError("Could not extract a repository from every PR URL")

# %% [markdown]
# ## Bug-labeled cohort composition
#
# The paper's initial 539 repositories are not the population represented by this
# table. That number was obtained after the initial AIDev language and popularity
# filters. Validation reduced it to 489 unique repositories with at least one
# qualifying agentic PR (Table 1). The RQ3 authors then retained merged PRs,
# balanced agentic and human PRs within repository quarters, and excluded quarters
# that contained only one group. Their released bug-labeled CSV contains 171
# repositories across all five agents. Our analysis additionally excludes Copilot
# and Devin, leaving 106 unique repositories for Claude, Codex, and Cursor. The
# paper does not provide a repository-level attrition table, so the exact reduction
# from 489 to 171 cannot be decomposed further from the paper alone.

# %%
bug_labels = pd.read_csv(bug_csv, usecols=["pr_url"])
bug_label_repositories = bug_labels["pr_url"].str.extract(PR_URL_RE)[0].str.lower()
human_prs = (rq3["pr_type"] == "HM").sum()
agentic_bot_prs = ((rq3["pr_type"] == "AM") & (rq3["pr_usertype"] == "Bot")).sum()
agentic_non_bot_prs = ((rq3["pr_type"] == "AM") & (rq3["pr_usertype"] != "Bot")).sum()

print("Released bug-label CSV")
print(f"  Total PRs across all five agents: {len(bug_labels):,}")
print(f"  Unique repositories across all five agents: {bug_label_repositories.nunique():,}")
print()
print("Bug-labeled cohort composition (Claude, Codex, and Cursor)")
print(f"  Total PRs: {len(rq3):,}")
print(f"  Human-generated PRs (HM): {human_prs:,}")
print(f"  Agentic PRs authored by bot accounts: {agentic_bot_prs:,}")
print(f"  Agentic PRs authored by non-bot accounts: {agentic_non_bot_prs:,}")
print(f"  Unique repositories: {rq3['repository'].nunique():,}")

# %% [markdown]
# ## Simplified interaction model with raw experience
#
# Logistic regression tests whether the association between prior merged PRs and bug-inducing status differs between PRs with and without AI. Dividing experience by 100 makes the coefficient the change in log-odds associated with 100 additional prior merged PRs. This specification assumes a linear effect of the raw PR count and uses conventional standard errors.

# %%
simple_df = rq3.copy()
simple_df["is_ai"] = simple_df["is_ai"].astype(int)
simple_df["is_bug"] = (simple_df["is_bug"] > 0).astype(int)

simple_df["experience"] = simple_df["prior_merged_prs"]
simple_df["experience_100"] = simple_df["experience"] / 100

simple_model = smf.logit(
    "is_bug ~ is_ai * experience_100",
    data=simple_df,
).fit(disp=False)

term = "is_ai:experience_100"
print("Interaction coefficient:", simple_model.params[term])
print("95% confidence interval:", simple_model.conf_int().loc[term].tolist())
print("p-value:", simple_model.pvalues[term])

# %% [markdown]
# **Interpretation.** The estimated interaction is $-0.192$ log-odds per 100 additional prior merged PRs (95% CI $[-0.475, 0.092]$, $p = 0.185$). Because the confidence interval includes zero, this simpler raw-count specification does not provide statistically significant evidence that the association between experience and bug-inducing status differs between PRs with and without AI.

# %% [markdown]
# ## Simplified interaction model with log-transformed experience
#
# This model changes only the experience variable from the preceding model. It uses $\ln(\text{prior merged PRs} + 1)$, while retaining the same binary outcome, interaction formula, and conventional standard errors. This makes the raw-count and log-transformed specifications directly comparable.

# %%
log_experience_df = rq3.copy()
log_experience_df["is_ai"] = log_experience_df["is_ai"].astype(int)
log_experience_df["is_bug"] = (log_experience_df["is_bug"] > 0).astype(int)
log_experience_df["experience"] = log_experience_df["log_prior_merged_prs"]

log_experience_model = smf.logit(
    "is_bug ~ is_ai * experience",
    data=log_experience_df,
).fit(disp=False)

term = "is_ai:experience"
print("Interaction coefficient:", log_experience_model.params[term])
print("95% confidence interval:", log_experience_model.conf_int().loc[term].tolist())
print("p-value:", log_experience_model.pvalues[term])

# %% [markdown]
# **Interpretation.** The estimated interaction is $-0.162$ log-odds per one-unit increase in $\ln(\text{prior merged PRs} + 1)$ (95% CI $[-0.333, 0.008]$, $p = 0.062$). The estimate is closer to the conventional significance threshold than the raw-count estimate, but its confidence interval still includes zero.

# %% [markdown]
# ## Visualize the interaction
#
# This plot uses the log-transformed model above. It shows the model-predicted probability that a PR is bug-inducing across the observed range of prior merged PR counts. The shaded areas are 95% confidence intervals for the mean predictions.

# %%
max_experience = int(log_experience_df["prior_merged_prs"].max())
experience_grid = np.unique(
    np.rint(np.geomspace(1, max_experience + 1, 250) - 1).astype(int)
)

fig, (ax, interaction_ax) = plt.subplots(
    1,
    2,
    figsize=(11, 5),
    gridspec_kw={"width_ratios": [3, 1]},
)
colors = {0: "#4C78A8", 1: "#E45756"}
labels = {0: "Without AI", 1: "With AI"}

for is_ai in (0, 1):
    prediction_data = pd.DataFrame(
        {
            "is_ai": is_ai,
            "experience": np.log1p(experience_grid),
        }
    )
    prediction = log_experience_model.get_prediction(
        prediction_data
    ).summary_frame(alpha=0.05)

    ax.plot(
        experience_grid,
        100 * prediction["predicted"],
        color=colors[is_ai],
        linewidth=2.5,
        label=labels[is_ai],
    )
    ax.fill_between(
        experience_grid,
        100 * prediction["ci_lower"],
        100 * prediction["ci_upper"],
        color=colors[is_ai],
        alpha=0.18,
    )

ax.set_xscale("symlog", linthresh=1)
ax.set(
    title="Predicted bug-inducing probability",
    xlabel="Prior merged PRs by the author in the repository (log scale)",
    ylabel="Predicted probability of a bug-inducing PR (%)",
)
ax.legend(title="PR type", frameon=False)
ax.grid(axis="y", alpha=0.25)

interaction_term = "is_ai:experience"
interaction_estimate = log_experience_model.params[interaction_term]
interaction_ci = log_experience_model.conf_int().loc[interaction_term]
interaction_pvalue = log_experience_model.pvalues[interaction_term]

interaction_ax.errorbar(
    interaction_estimate,
    0,
    xerr=[
        [interaction_estimate - interaction_ci.iloc[0]],
        [interaction_ci.iloc[1] - interaction_estimate],
    ],
    fmt="o",
    color="#725AC1",
    markersize=8,
    capsize=6,
    linewidth=2,
)
interaction_ax.axvline(0, color="black", linestyle="--", linewidth=1)
interaction_ax.set(
    title=f"Interaction term\np = {interaction_pvalue:.3f}",
    xlabel="Coefficient (log-odds)\nwith 95% CI",
    yticks=[],
)
interaction_ax.grid(axis="x", alpha=0.25)

fig.suptitle("AI × Experience Interaction", fontsize=14)
fig.tight_layout()
plt.show()

# %% [markdown]
# **How to read the plot.** In the left panel, the vertical gap between the lines is the modeled AI/non-AI difference at a given experience level. Each line's slope shows the modeled association between experience and bug-inducing probability for that PR type. The separate confidence bands should not be used to test the interaction. The right panel shows the formal interaction estimate and its 95% confidence interval. The interval crosses zero and the interaction is not statistically significant at the 0.05 level ($p = 0.062$).
