import pytest

def test_audit_data_all_pass(custom_client, df_ok, common_kwargs):
    all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
        df=df_ok,
        **common_kwargs
    )
    assert all_pass
    assert len(fail_dfs) == 0
    assert len(case_specific_dfs) == 0

def test_audit_data_with_duplicates(custom_client, df_with_duplicates, common_kwargs):
    all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
        df=df_with_duplicates,
        **common_kwargs
    )
    assert not all_pass
    assert len(case_specific_dfs) == 0
    assert len(fail_dfs) == 2
    assert 'D001' in fail_dfs
    # The two duplicate rows should be returned
    assert len(fail_dfs['D001']) == 2
    assert 'D002' in fail_dfs
    ## D002 can not be run with duplicates
    assert fail_dfs["D002"] is None

def test_clean_data_with_duplicates(custom_client, df_with_duplicates, common_kwargs):
    all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
        df=df_with_duplicates,
        **common_kwargs
    )
    cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
        df=df_with_duplicates,
        fail_dict=fail_dfs, case_specific_dict=case_specific_dfs,
        agg_dict={'y': 'sum'}, **common_kwargs
    )
    assert all_pass
    assert len(fail_dfs) == 0
    assert len(case_specific_dfs) == 0
    assert len(cleaned_df) == 3

def test_clean_data_raises_valueerror(custom_client, df_with_duplicates, common_kwargs):
    _, fail_dfs, case_specific_dfs = custom_client.audit_data(
        df=df_with_duplicates,
        **common_kwargs
    )
    with pytest.raises(ValueError, match="agg_dict must be provided to resolve D001 failure."):
        custom_client.clean_data(
            df=df_with_duplicates,
            fail_dict=fail_dfs,
            case_specific_dict=case_specific_dfs,
            **common_kwargs
        )