from datetime import datetime
from functools import partial
from typing import Callable, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
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
    COLORS = {
        "predicted": "#51A5BA",
        "actual": "#6BCAE2",
    }

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
        self.monthly_df = None

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
        self.df = self.df_raw.groupby("date").agg({"consumption": "sum", "temperature": "mean"})

        if self.dateranges_to_group_to is None:
            # use the whole date range as the default group
            self.dateranges_to_group_to = [DateRange(self.df.index.min(), self.df.index.max())]

        self.baseline_df = self.apply_daterange(self.df, date_range=self.baseline_period)
        self._fit_function_to_baseline_period()

        fitted_function = bind(
            self.fitting_function,
            ...,
            *self.fitting_function_coefficients,
        )
        self.df["baseline_consumption"] = self.df["temperature"].apply(fitted_function)
        self.df["saved_daily_consumption"] = self.df.apply(
            lambda row: row["baseline_consumption"] - row["consumption"],
            axis=1,
        )
        self.df["cumulative_consumption"] = self.df["consumption"].cumsum()
        self.df["cumulative_predicted_consumption"] = self.df["baseline_consumption"].cumsum()

        self.df["saved_cumulative_consumption"] = self.df["saved_daily_consumption"].cumsum()

        # create a new column "month" so that a new monthly aggregated dataframe can be added
        self.df["month"] = self.df.index
        # Convert to datetime
        self.df["month"] = pd.to_datetime(self.df["month"])
        # format to string YYYY-mm
        self.df["month"] = self.df["month"].dt.strftime("%Y-%m")

        # Create a new dataframe with monthly aggregated data
        self.monthly_df = self.df.groupby("month").agg(
            {
                "consumption": "sum",
                "baseline_consumption": "sum",
                "temperature": "mean",
            }
        )
        self.monthly_df["month"] = self.monthly_df.index
        self.monthly_df["month_dt"] = pd.to_datetime(self.monthly_df["month"])
        self.monthly_df["in_baseline_period"] = self.monthly_df["month_dt"].apply(
            self.baseline_period.timestamp_in_date_range
        )

        self.monthly_df["diff"] = self.monthly_df["consumption"] - self.monthly_df["baseline_consumption"]

        self.monthly_df["diff_percents"] = (
            100
            * (self.monthly_df["consumption"] - self.monthly_df["baseline_consumption"])
            / self.monthly_df["baseline_consumption"]
        )

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
        legend.append(f"Baseline consumption {self.baseline_period.begin} - {self.baseline_period.end}")

        ax.legend(legend)

    def create_plot_cumnulative_saved_electricity_vs_time(self, ax: plt.axes):
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative saved electricity [kWh]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        begin_date = self.apply_daterange(
            self.df,
            date_range=self.dateranges_to_group_to[0],
        ).index[0]
        initial_saved_cumulative_consumption = self.df["saved_cumulative_consumption"][begin_date]

        legend = []
        for date_range in self.dateranges_to_group_to:
            df_daterange = self.apply_daterange(self.df, date_range=date_range)
            x_data = df_daterange.index
            y_data = df_daterange["saved_cumulative_consumption"] - initial_saved_cumulative_consumption
            ax.scatter(x_data, y_data, s=self.dot_size, color=date_range.color)
            legend.append(
                f"Cumulative saved electricity compared to baseline [kWh] {date_range.begin} - {date_range.end}"
            )

    def create_plot_cumnulative_saved_electricity_percents(self, ax: plt.axes):
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative saved electricity [%]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        begin_date = self.apply_daterange(
            self.df,
            date_range=self.dateranges_to_group_to[0],
        ).index[0]
        initial_cumulative_consumption = self.df["cumulative_consumption"][begin_date]
        initial_saved_cumulative_consumption = self.df["saved_cumulative_consumption"][begin_date]

        legend = []
        for date_range in self.dateranges_to_group_to:
            df_daterange = self.apply_daterange(self.df, date_range=date_range)
            x_data = df_daterange.index
            y_data = (
                (df_daterange["saved_cumulative_consumption"] - initial_saved_cumulative_consumption)
                / (df_daterange["cumulative_consumption"] - initial_cumulative_consumption)
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
            self.df,
            date_range=self.dateranges_to_group_to[0],
        ).index[0]
        initial_cumulative_consumption = self.df["cumulative_consumption"][begin_date]

        legend = []
        for date_range in self.dateranges_to_group_to:
            df_daterange = self.apply_daterange(self.df, date_range=date_range)
            x_data = df_daterange.index
            y_data = df_daterange["cumulative_consumption"] - initial_cumulative_consumption
            ax.scatter(x_data, y_data, s=self.dot_size, color=date_range.color)
            legend.append(f"Cumulative consumed electricity [kWh] {date_range.begin} - {date_range.end}")

    def create_plot_monthly_bar_chart(self, ax: plt.axes):
        """
        Visualize monthly predicted consumption vs actual consumption as a bar chart
        """
        self.monthly_df.plot.bar(
            x="month",
            y=["baseline_consumption", "consumption"],
            ax=ax,
            color=[
                self.COLORS["predicted"],
                self.COLORS["actual"],
            ],
        )
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(["Predicted consumption", "Actual consumption"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Consumed electricity [kWh]")

        patches_len = len(ax.patches)
        for i, bar in enumerate(ax.patches[patches_len // 2 :]):
            # get the x and y coordinates of the bar
            x = bar.get_x()
            baseline_consumption = self.monthly_df["baseline_consumption"][i]
            consumption = self.monthly_df["consumption"][i]

            if self.monthly_df["in_baseline_period"][i] or np.isnan(self.monthly_df["diff_percents"][i]):
                continue

            reduction = round(self.monthly_df["diff_percents"][i], 2)
            diff = int(round(self.monthly_df["diff"][i], 0))

            sign = ""
            if reduction >= 0:
                sign = "+"
            sign = "+" if reduction >= 0 else ""
            color = "red" if reduction > 0 else "green"
            ax.text(
                x,
                max([baseline_consumption, consumption]),
                f"{sign}{diff} kWh \n{sign}{reduction} %",
                ha="center",
                va="bottom",
                color=color,
            )

        # baseline period boundaries
        baseline_begin_index = self.monthly_df[self.monthly_df["in_baseline_period"] == True].index[0]
        baseline_end_index = self.monthly_df[self.monthly_df["in_baseline_period"] == True].index[-1]
        baseline_begin = self.monthly_df.index.get_loc(baseline_begin_index)
        baseline_end = self.monthly_df.index.get_loc(baseline_end_index) + 1 - baseline_begin

        # # Add a shaded background to the baseline period
        rect = Rectangle(
            xy=(baseline_begin - 0.5, 0), width=baseline_end, height=ax.get_ylim()[1], facecolor="#F0F0F0", zorder=-1
        )
        ax.add_patch(rect)

        # Add the floating text
        ax.text(8 / 2, ax.get_ylim()[1] * 0.9, "Baseline period", ha="center", va="center", fontsize=14, color="black")

        ax2 = ax.twinx()
        ax2.plot(self.monthly_df["month"], self.monthly_df["temperature"], label="Temperature", color="red")
        ax2.set_ylabel("Average outside temperature [°C]", color="red")
