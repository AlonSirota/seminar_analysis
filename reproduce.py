# %% [markdown]
# # Developer Experience vs. Bug-Inducing Rate in Agentic PRs
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
    rq3["is_agentic"] = (rq3["pr_type"] == "AM").astype(int)

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

    rq3_ext = rq3["pr_url"].str.extract(PR_URL_RE)
    rq3["pr_key"] = rq3_ext[0].str.lower() + "#" + rq3_ext[1]
    rq3 = rq3.merge(
        unique_prs[["pr_key", "prior_merged_prs"]],
        on="pr_key",
        how="left",
    )
    rq3["is_bug"] = np.where(rq3["is_bug_inducing"] == "BI", 100.0, 0.0)
    rq3.to_csv(cohort_csv, index=False)

rq3 = pd.read_csv(cohort_csv)
rq3["is_agentic"] = (rq3["pr_type"] == "AM").astype(int)
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
# ## Interaction model with raw experience
#
# Logistic regression tests whether the association between prior merged PRs and bug-inducing status differs between agentic and human PRs. Experience is the raw number of prior merged PRs, so its coefficient is the change in log-odds associated with one additional prior merged PR. This specification assumes a linear effect of the raw PR count and uses conventional standard errors.

# %%
simple_df = rq3.copy()
simple_df["is_agentic"] = simple_df["is_agentic"].astype(int)
simple_df["is_bug"] = (simple_df["is_bug"] > 0).astype(int)

simple_df["experience"] = simple_df["prior_merged_prs"]

raw_experience_model = smf.logit(
    "is_bug ~ is_agentic * experience",
    data=simple_df,
).fit(disp=False)

term = "is_agentic:experience"
print("Interaction coefficient:", raw_experience_model.params[term])
print("95% confidence interval:", raw_experience_model.conf_int().loc[term].tolist())
print("p-value:", raw_experience_model.pvalues[term])

# %% [markdown]
# **Interpretation.** The estimated interaction is $-0.0019$ log-odds per additional prior merged PR (95% CI $[-0.0048, 0.0009]$, $p = 0.185$). Because the confidence interval includes zero, the raw-count specification does not provide statistically significant evidence that the association between experience and bug-inducing status differs between agentic and human PRs.

# %% [markdown]
# ## Predicted probabilities for agentic and human PRs
#
# The plot shows predictions from the raw-count interaction model with 95% confidence intervals. It stops at the largest experience value observed in both cohorts, avoiding comparisons outside their common observed range.

# %%
max_experience = int(simple_df.groupby("is_agentic")["prior_merged_prs"].max().min())
experience_grid = np.linspace(0, max_experience, 250)

fig, ax = plt.subplots(figsize=(7, 5))
colors = {0: "#4C78A8", 1: "#E45756"}
human_slope_pvalue = raw_experience_model.pvalues["experience"]
agentic_slope_pvalue = raw_experience_model.t_test(
    "experience + is_agentic:experience = 0"
).pvalue.item()
labels = {
    0: f"Human PRs (slope p = {human_slope_pvalue:.3f})",
    1: f"Agentic PRs (slope p = {agentic_slope_pvalue:.3f})",
}

for is_agentic in (0, 1):
    prediction_data = pd.DataFrame(
        {
            "is_agentic": is_agentic,
            "experience": experience_grid,
        }
    )
    prediction = raw_experience_model.get_prediction(prediction_data).summary_frame(alpha=0.05)

    ax.plot(
        experience_grid,
        100 * prediction["predicted"],
        color=colors[is_agentic],
        linewidth=2.5,
        label=labels[is_agentic],
    )
    ax.fill_between(
        experience_grid,
        100 * prediction["ci_lower"],
        100 * prediction["ci_upper"],
        color=colors[is_agentic],
        alpha=0.18,
    )

interaction_pvalue = raw_experience_model.pvalues["is_agentic:experience"]
ax.set(
    title=f"PR Type × Experience Model\ninteraction p = {interaction_pvalue:.3f}",
    xlabel="Prior merged PRs in the repository",
    ylabel="Predicted probability of a bug-inducing PR (%)",
)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="PR type", frameon=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# **How to read the plot.** At a given experience level, the vertical gap between the lines is the modeled difference between agentic and human PRs. Each legend p-value tests whether that cohort's experience slope differs from zero. The interaction p-value tests whether the two slopes differ from each other. Neither cohort slope is statistically significant (human PRs: $p = 0.615$; agentic PRs: $p = 0.071$), and the difference between the slopes is also not statistically significant ($p = 0.185$).
