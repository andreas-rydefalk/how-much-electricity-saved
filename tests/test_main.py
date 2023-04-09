import datetime

import pandas as pd
import pytest

from how_much_electricity_saved import fitting_functions
from how_much_electricity_saved.date_range import DateRange
from how_much_electricity_saved.main import ElectricityConsumptionReport


@pytest.fixture
def sample_report() -> ElectricityConsumptionReport:
    data = {
        "time": [
            "2022-01-01 00:00:00",
            "2022-01-02 00:00:00",
            "2022-01-03 00:00:00",
        ],
        "consumption": [
            10,
            9,
            8,
        ],
        "temperature": [1, 2, 3],
    }
    df = pd.DataFrame(data)

    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")

    report = ElectricityConsumptionReport(
        dataframe=df,
        baseline_period=DateRange(begin="2022-01-01", end="2022-02-01"),
        fitting_function=fitting_functions.fitting_func_linear,
    )
    return report


def test_init(sample_report: ElectricityConsumptionReport):
    assert "consumption" in sample_report.df.columns
    assert "temperature" in sample_report.df.columns
    assert sample_report.baseline_period.begin == datetime.date(2022, 1, 1)
    assert sample_report.baseline_period.end == datetime.date(2022, 2, 1)
    assert sample_report.fitting_function == fitting_functions.fitting_func_linear


@pytest.mark.parametrize(
    "required_col",
    [
        "baseline_consumption",
        "saved_daily_consumption",
        "cumulative_consumption",
        "cumulative_predicted_consumption",
        "saved_cumulative_consumption",
        "month",
    ],
)
def test_preprocess_assert_df_columns(sample_report: ElectricityConsumptionReport, required_col):
    assert required_col in sample_report.df.columns


@pytest.mark.parametrize(
    "required_col",
    [
        "consumption",
        "baseline_consumption",
        "temperature",
        "month",
        "month_dt",
        "in_baseline_period",
        "diff",
        "diff_percents",
    ],
)
def test_preprocess_assert_monthly_df_columns(sample_report: ElectricityConsumptionReport, required_col):
    assert required_col in sample_report.monthly_df.columns


@pytest.mark.parametrize(
    "required_col",
    [
        "consumption",
        "temperature",
    ],
)
def test_preprocess_assert_baseline_df_columns(sample_report: ElectricityConsumptionReport, required_col):
    assert required_col in sample_report.baseline_df.columns
