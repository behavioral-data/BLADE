import pytest
from blade_bench.eval.datamodel.lm_analysis import ModelAndColumns
from blade_bench.eval.llm import CodeToTransformsLLM, CodeToModelLLM


def test_code_to_model():
    llm = CodeToModelLLM.init_from_base_llm_config()
    code_snippet = """
fm1 <- lm(score34 ~ stratio, data = MASchools)
coeftest(fm1, vcov = vcovHC(fm1, type = "HC1"))
    """
    r1 = llm.code_to_model(code_snippet)
    r1_obj = llm.code_to_model_obj(code_snippet)
    assert isinstance(r1, str)
    assert isinstance(r1_obj, ModelAndColumns)

    code_snippet = """
fm_probit2 <- glm(model, data = MurderRates , family = binomial (link = "probit"),
control = list(epsilon = 1e-15, maxit = 50, trace = FALSE))
summary(fm_probit2)
    """
    r2 = llm.code_to_model(code_snippet)
    r2_obj = llm.code_to_model_obj(code_snippet)
    assert isinstance(r2, str)
    assert isinstance(r2_obj, ModelAndColumns)


def test_code_to_model_obj():
    llm = CodeToTransformsLLM.init_from_base_llm_config()
    code_example = "df[df['Female_Contributions'] == df['UniqueFemaleContributors']].groupby('ThreadId').apply(lambda grp: grp.assign(WC_s=scale(grp['WC'])))"
    s = llm.convert_code(code_example)
    assert isinstance(s, str)

    code_example = """
df = df[df['Female_Contributions'] == df['UniqueFemaleContributors']]
df = df.groupby('ThreadId').apply(lambda group: group.assign(row_number=range(1, len(group) + 1))).reset_index(drop=True)
df = df.groupby(['Title']).apply(lambda group: group.iloc[0]).reset_index(drop=True)
df = df[df['Female'] == 1]
df = df.groupby(['Title', 'Id']).agg(
    co        prev_threads_mean=('PreviousThreads', 'mean'),
    threads_this_year_mean=('ThreadsThisYear'mments=('Id', 'count'),
, 'mean'),
    cont_this_year_mean=('ContributionsThisYear', 'mean'),
    prev_contributions_mean=('PreviousContributions', 'mean')
).reset_index()
df['total_prev_threads'] = df['prev_threads_mean'] + df['threads_this_year_mean']
df['total_prev_contributions'] = df['cont_this_year_mean'] + df['prev_contributions_mean']
df['cont_per_thread'] = df['total_prev_contributions'] / df['total_prev_threads']
df['comments_now_percent_change'] = (df['comments'] - df['cont_per_thread']) * 100 / df['cont_per_thread']
    """
    s2 = llm.convert_code(code_example)
    assert isinstance(s2, str)

    code_example = """
df = (df[df['Female_Contributions'] == df['UniqueFemaleContributors']]
          .groupby('ThreadId')
          .apply(lambda x: x.sort_values(by='Order'))
          .reset_index(drop=True))
df['next_female'] = df['Female'].shift(-1)
    """
    s3 = llm.convert_code(code_example)
    assert isinstance(s3, str)


if __name__ == "__main__":
    pytest.main([__file__])
