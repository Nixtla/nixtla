import pandas as pd

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

def test_audit_data_with_missing_dates(custom_client, df_with_missing_dates, common_kwargs):
    all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
        df=df_with_missing_dates,
        **common_kwargs
    )
    assert not all_pass
    assert len(case_specific_dfs) == 0
    assert len(fail_dfs) == 1
    assert 'D002' in fail_dfs
    assert len(fail_dfs['D002']) == 2  # Two missing dates should be returned

def test_clean_data_with_missing_dates(custom_client, df_with_missing_dates, common_kwargs):
    # First audit to get fail_dfs and case_specific_dfs
    _, fail_dfs, case_specific_dfs = custom_client.audit_data(
        df=df_with_missing_dates,
        **common_kwargs
    )
    cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
        df=df_with_missing_dates,
        fail_dict=fail_dfs, case_specific_dict=case_specific_dfs,
        agg_dict={'y': 'sum'},
        **common_kwargs
    )
    assert all_pass
    assert len(fail_dfs) == 0
    assert len(case_specific_dfs) == 0
    assert len(cleaned_df) == 6  # Two missing rows added.
    assert pd.to_datetime("2023-01-02") in pd.to_datetime(cleaned_df['ds']).values

def test_audit_data_with_duplicates_and_missing_dates(custom_client, df_with_duplicates_and_missing_dates, common_kwargs):
    all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
        df=df_with_duplicates_and_missing_dates,
        **common_kwargs
    )
    assert not all_pass
    assert len(case_specific_dfs) == 0
    assert len(fail_dfs) == 2
    assert 'D001' in fail_dfs
    assert len(fail_dfs['D001']) == 2  # The two duplicate rows should be returned
    assert 'D002' in fail_dfs
    assert fail_dfs["D002"] is None  # D002 can not be run with duplicates

def test_clean_data_with_duplicates_and_missing_dates(custom_client, df_with_duplicates_and_missing_dates, common_kwargs):
    # First audit to get fail_dfs and case_specific_dfs
    _, fail_dfs, case_specific_dfs = custom_client.audit_data(
        df=df_with_duplicates_and_missing_dates,
        **common_kwargs
    )

    # Clean Data (pass 1 will clear the duplicates)
    cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
        df=df_with_duplicates_and_missing_dates,
        fail_dict=fail_dfs, case_specific_dict=case_specific_dfs,
        agg_dict={'y': 'sum'}, **common_kwargs
    )
    assert not all_pass
    assert len(fail_dfs) == 1
    # Since duplicates have been removed, D002 has been run now.
    assert 'D002' in fail_dfs
    assert len(fail_dfs["D002"]) == 1
    assert len(case_specific_dfs) == 0
    assert len(cleaned_df) == 4  # Two duplicates rows consolidated into one.

    # Clean Data (pass 2 will clear the missing dates)
    cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
        df=cleaned_df,
        fail_dict=fail_dfs, case_specific_dict=case_specific_dfs,
        **common_kwargs
    )
    assert all_pass
    assert len(fail_dfs) == 0
    assert len(case_specific_dfs) == 0
    # Two duplicates rows consolidated into one plus one missing row added.
    assert len(cleaned_df) == 5
