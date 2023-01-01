from functools import partial
from typing import Callable, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from how_much_electricity_saved.date_range import DateRange


class MissingColumnException(ValueError):
    pass


class bind(partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder
    https://stackoverflow.com/a/66274908
    """

    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)


class ElectricityConsumptionReport:

    required_dataframe_columns = ["time", "consumption", "temperature"]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        baseline_period: DateRange,
        fitting_function: Callable,
        initial_guess: Iterable = None,
        dateranges_to_group_to: Iterable[DateRange] = None,
        dot_size: int = 10,
        _preprocess_at_init: bool = True,
    ) -> None:
        """
        Estimate how much electricity has been saved compared to baseline consumption.

        # Parameters:
        dataframe: pd.DataFrame
            Required columns:
               - time
               - consumption
               - temperature (= outside temperature)
        baseline_period: DataRange
            Defines the baseline period. A curve is fitted to consumption data during
            this period in order to get a function that models how electricity consumption is
            changing as a function of temperature.
        fitting_function: Callable,
            The model function, f(x, ...). It must take the independent variable as the first argument
            and the parameters to fit as separate remaining arguments.
        initial_guess: Iterable = None,
            An initial guess for the fitting function
        dateranges_to_group_to: Iterable[DateRange] = None
            Optional grouping of data when visualized.
        dot_size: int = 10
            Optionally adjust the dot size in the scatter plot
        """
        if not all(col in dataframe.columns for col in self.required_dataframe_columns):
            raise MissingColumnException(
                (
                    "One or more column is missing from the"
                    "dataframe. Required columns are: "
                    f"{self.required_dataframe_columns}"
                )
            )
        self.df_raw = dataframe.copy()
        self.df = None
        self.baseline_period = baseline_period
        self.fitting_function = fitting_function
        self.initial_guess = initial_guess
        self.dateranges_to_group_to = dateranges_to_group_to

        # Initialize these to None. Will be filled in preprocessing
        self.baseline_df = None
        self.fitting_function_coefficients = None
        self.covariance_matrix = None

        self.dot_size = dot_size
        if _preprocess_at_init:
            self._preprocess()

    def _preprocess(self) -> None:
        """
        Data preprocessing
        """
        # Create a new column with just the date
        self.df_raw["date"] = self.df_raw["time"].dt.date

        # Group the data by date and calculate the sum of the hourly_consumption and the average of the outside_temperature
        self.df = self.df_raw.groupby("date").agg(
            {"consumption": "sum", "temperature": "mean"}
        )

        if self.dateranges_to_group_to is None:
            # use the whole date range as the default group
            self.dateranges_to_group_to = [
                DateRange(self.df.index.min(), self.df.index.max())
            ]

        self.baseline_df = self.apply_daterange(
            self.df, date_range=self.baseline_period
        )
        self._fit_function_to_baseline_period()

        fitted_function = bind(
            self.fitting_function, ..., *self.fitting_function_coefficients
        )
        self.df["baseline_consumption"] = self.df["temperature"].apply(fitted_function)
        self.df["saved_daily_consumption"] = self.df.apply(
            lambda row: row["baseline_consumption"] - row["consumption"], axis=1
        )
        self.df["cumulative_consumption"] = self.df["consumption"].cumsum()

        self.df["saved_cumulative_consumption"] = self.df[
            "saved_daily_consumption"
        ].cumsum()

    @staticmethod
    def apply_daterange(df, date_range: DateRange) -> pd.DataFrame:
        return df.loc[date_range.begin : date_range.end]

    def _fit_function_to_baseline_period(self) -> None:

        x_data = self.baseline_df["temperature"]
        y_data = self.baseline_df["consumption"]

        if self.initial_guess is not None:
            self.fitting_function_coefficients, self.covariance_matrix = curve_fit(
                self.fitting_function, xdata=x_data, ydata=y_data, p0=self.initial_guess
            )
        else:
            self.fitting_function_coefficients, self.covariance_matrix = curve_fit(
                self.fitting_function,
                xdata=x_data,
                ydata=y_data,
            )

    def create_scatter_plot_total(self, ax: plt.axes):

        ax.set_xlabel("Outside temperature [°C]")
        ax.set_ylabel("Electicity consumption [kWh/24h]")
        legend = []
        for date_range in self.dateranges_to_group_to:
            df_daterange = self.apply_daterange(self.df, date_range=date_range)
            x_data = df_daterange["temperature"]
            y_data = df_daterange["consumption"]
            ax.scatter(x_data, y_data, s=self.dot_size, color=date_range.color)
            legend.append(f"Consumption {date_range.begin} - {date_range.end}")

        temperature_max = self.baseline_df["temperature"].max()
        temperature_min = self.baseline_df["temperature"].min()
        temperature_x = np.linspace(temperature_min, temperature_max, 100)
        ax.plot(
            temperature_x,
            self.fitting_function(temperature_x, *self.fitting_function_coefficients),
            color="green",
        )
        legend.append(
            f"Baseline consumption {self.baseline_period.begin} - {self.baseline_period.end}"
        )

        ax.legend(legend)

    def create_plot_cumnulative_saved_electricity_vs_time(self, ax: plt.axes):

        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative saved electricity [kWh]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        begin_date = self.apply_daterange(
            self.df, date_range=self.dateranges_to_group_to[0]
        ).index[0]
        initial_saved_cumulative_consumption = self.df["saved_cumulative_consumption"][
            begin_date
        ]

        legend = []
        for date_range in self.dateranges_to_group_to:
            df_daterange = self.apply_daterange(self.df, date_range=date_range)
            x_data = df_daterange.index
            y_data = (
                df_daterange["saved_cumulative_consumption"]
                - initial_saved_cumulative_consumption
            )
            ax.scatter(x_data, y_data, s=self.dot_size, color=date_range.color)
            legend.append(
                f"Cumulative saved electricity compared to baseline [kWh] {date_range.begin} - {date_range.end}"
            )

    def create_plot_cumnulative_saved_electricity_percents(self, ax: plt.axes):

        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative saved electricity [%]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        begin_date = self.apply_daterange(
            self.df, date_range=self.dateranges_to_group_to[0]
        ).index[0]
        initial_cumulative_consumption = self.df["cumulative_consumption"][begin_date]
        initial_saved_cumulative_consumption = self.df["saved_cumulative_consumption"][
            begin_date
        ]

        legend = []
        for date_range in self.dateranges_to_group_to:
            df_daterange = self.apply_daterange(self.df, date_range=date_range)
            x_data = df_daterange.index
            y_data = (
                (
                    df_daterange["saved_cumulative_consumption"]
                    - initial_saved_cumulative_consumption
                )
                / (
                    df_daterange["cumulative_consumption"]
                    - initial_cumulative_consumption
                )
                * 100
            )
            ax.scatter(x_data, y_data, s=self.dot_size, color=date_range.color)
            legend.append(
                f"Cumulative saved electricity compared to baseline [%] {date_range.begin} - {date_range.end}"
            )

    def create_plot_cumnulative_consumed_electricity(self, ax: plt.axes):

        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative consumed electricity [kWh]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        begin_date = self.apply_daterange(
            self.df, date_range=self.dateranges_to_group_to[0]
        ).index[0]
        initial_cumulative_consumption = self.df["cumulative_consumption"][begin_date]

        legend = []
        for date_range in self.dateranges_to_group_to:
            df_daterange = self.apply_daterange(self.df, date_range=date_range)
            x_data = df_daterange.index
            y_data = (
                df_daterange["cumulative_consumption"] - initial_cumulative_consumption
            )
            ax.scatter(x_data, y_data, s=self.dot_size, color=date_range.color)
            legend.append(
                f"Cumulative consumed electricity [kWh] {date_range.begin} - {date_range.end}"
            )
