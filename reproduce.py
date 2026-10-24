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

    rq3 = bug_df.merge(info_df, on=["pr_url", "pr_type", "agent"])
    rq3 = rq3[rq3["agent"].isin(AGENTS)].copy()

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

    matches = all_prs["pr_url"].str.extract(PR_URL_RE)
    all_prs["pr_key"] = matches[0].str.lower() + "#" + matches[1]
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

    unique_prs["prior_merged_prs"] = count_prior_merged_prs(unique_prs)

    matches = rq3["pr_url"].str.extract(PR_URL_RE)
    rq3["pr_key"] = matches[0].str.lower() + "#" + matches[1]
    rq3 = rq3.merge(
        unique_prs[["pr_key", "prior_merged_prs"]],
        on="pr_key",
        how="left",
    )
    rq3["is_bug"] = (rq3["is_bug_inducing"] == "BI").astype(int)
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
bug_repos = bug_labels["pr_url"].str.extract(PR_URL_RE)[0].str.lower()
agentic = rq3["pr_type"] == "AM"
bots = rq3["pr_usertype"] == "Bot"

print("Released bug-label CSV")
print(f"  Total PRs across all five agents: {len(bug_labels):,}")
print(f"  Unique repositories across all five agents: {bug_repos.nunique():,}")
print()
print("Bug-labeled cohort composition (Claude, Codex, and Cursor)")
print(f"  Total PRs: {len(rq3):,}")
print(f"  Human-generated PRs (HM): {(~agentic).sum():,}")
print(f"  Agentic PRs authored by bot accounts: {(agentic & bots).sum():,}")
print(f"  Agentic PRs authored by non-bot accounts: {(agentic & ~bots).sum():,}")
print(f"  Unique repositories: {rq3['repository'].nunique():,}")

# %% [markdown]
# ## Interaction model with raw experience
#
# Logistic regression tests whether the association between prior merged PRs and bug-inducing status differs between agentic and human PRs. Experience is the raw number of prior merged PRs, so its coefficient is the change in log-odds associated with one additional prior merged PR. This specification assumes a linear effect of the raw PR count and uses conventional standard errors.

# %%
analysis = rq3.assign(
    is_bug=(rq3["is_bug"] > 0).astype(int),
    experience=rq3["prior_merged_prs"],
)
model = smf.logit(
    "is_bug ~ is_agentic * experience",
    data=analysis,
).fit(disp=False)

term = "is_agentic:experience"

# %% [markdown]
# ## Experience estimates by PR type
#
# The first two rows estimate the relationship between experience and bug-inducing status within each PR type. The interaction tests whether those relationships differ. Estimates are changes in log-odds per additional prior merged PR.

# %%
agentic_experience = model.t_test("experience + is_agentic:experience = 0")
experience_results = pd.DataFrame(
    [
        ["Human PRs", model.params["experience"], model.pvalues["experience"]],
        ["Agentic PRs", agentic_experience.effect.item(), agentic_experience.pvalue.item()],
        ["PR type × experience", model.params[term], model.pvalues[term]],
    ],
    columns=["Result", "Estimate (log-odds)", "p-value"],
).set_index("Result")

experience_results.style.format({"Estimate (log-odds)": "{:.4f}", "p-value": "{:.3f}"})

# %% [markdown]
# **Interpretation.** For human PRs, prior experience has almost no estimated association with bug-inducing status ($p = 0.615$). For agentic PRs, greater experience is associated with a larger estimated reduction, although this association is not statistically significant at the 0.05 level ($p = 0.071$). The interaction is also not statistically significant ($p = 0.185$), so there is insufficient evidence that the relationship differs between human and agentic PRs.

# %% [markdown]
# ## Predicted probabilities for agentic and human PRs
#
# The plot shows predictions from the raw-count interaction model with 95% confidence intervals. It stops at the 95th percentile of experience because a few unusually high values would otherwise stretch the graph.

# %%
experience_grid = np.linspace(0, analysis["experience"].quantile(0.95))

fig, ax = plt.subplots(figsize=(7, 5))
for is_agentic, name in [(0, "Human PRs"), (1, "Agentic PRs")]:
    plot_data = pd.DataFrame({"is_agentic": is_agentic, "experience": experience_grid})
    pred = model.get_prediction(plot_data).summary_frame()

    (line,) = ax.plot(
        experience_grid,
        100 * pred["predicted"],
        linewidth=2.5,
        label=name,
    )
    ax.fill_between(
        experience_grid,
        100 * pred["ci_lower"],
        100 * pred["ci_upper"],
        color=line.get_color(),
        alpha=0.18,
    )

ax.set(
    title="Predicted bug-inducing probability by prior experience",
    xlabel="Prior merged PRs in the repository",
    ylabel="Predicted probability of a bug-inducing PR (%)",
)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="PR type", frameon=False)
fig.text(
    0.5,
    0.01,
    f"PR type × prior experience interaction: p = {model.pvalues[term]:.3f}",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
plt.show()

# %% [markdown]
# **How to read the plot.** At a given experience level, the vertical gap between the lines is the modeled difference between agentic and human PRs. The shaded areas are 95% confidence intervals. The interaction tests whether the relationship between prior experience and bug-inducing status differs by PR type. It is not statistically significant ($p = 0.185$).
