# %% [markdown]
# # Developer Experience vs. Bug-Inducing Rate in AI-Assisted PRs
#
# Secondary analysis on the RQ3 dataset from Mazloomzadeh et al. ([arXiv:2607.21832](https://arxiv.org/abs/2607.21832), [GitHub repo](https://github.com/mazloomzadeh/AI-Agent-Analysis)).
#
# Downloaded files and the merged cohort table are cached in `data_cache/`. Delete a file there if you want to re-download or rebuild it.

# %%
from pathlib import Path
import urllib.request
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
print(f"N = {len(rq3)} PRs (Human: {(rq3['is_ai'] == 0).sum()}, AI: {(rq3['is_ai'] == 1).sum()})")

# %% [markdown]
# ## 3. OLS interaction regressions
#
# Model: $\text{is\_bug} = \beta_0 + \beta_1\,\text{is\_ai} + \beta_2\,\text{Exp} + \beta_3\,(\text{is\_ai} \times \text{Exp})$
# - Human slope: $\beta_2$
# - AI slope: $\beta_2 + \beta_3$
# - Interaction (AI $-$ Human): $\beta_3$

# %%
metrics = {
    "log_prior_merged_prs": "Prior Merged PRs (ln(PRs + 1))",
    "log_days_first_merged": "Active Tenure (ln(Days + 1))",
}

rows = []
for col, label in metrics.items():
    m = smf.ols(f"is_bug ~ is_ai * {col}", data=rq3).fit()
    t_ai = m.t_test(f"{col} + is_ai:{col} = 0")
    inter = f"is_ai:{col}"

    rows.append(
        {
            "Metric": label,
            "Human Slope": f"{m.params[col]:+.3f} (p={m.pvalues[col]:.4f})",
            "AI Slope": f"{t_ai.effect.item():+.3f} (p={t_ai.pvalue.item():.4f})",
            "Interaction": f"{m.params[inter]:+.3f} (p={m.pvalues[inter]:.4f})",
        }
    )

print(pd.DataFrame(rows).to_string(index=False))
