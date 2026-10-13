tcga_goals = """
Your goal is to make scientific discoveries."""

# Closed task (experiment 06): the same data, but the agents test three prespecified
# hypotheses instead of exploring freely.
kirc_closed_hypotheses_goals = """
Using the supplied TCGA kidney renal clear cell carcinoma (KIRC) data, test whether the data provide
evidence supporting potential therapeutic benefit from inhibition of (1) ABL1 signaling, (2) CDK4/6,
and (3) JAK signaling. Treat these as prespecified hypotheses to evaluate rigorously; report
positive, negative, and contradictory evidence."""

GOALS = {
    "tcga": tcga_goals,
    "kirc_closed_hypotheses": kirc_closed_hypotheses_goals,
}
