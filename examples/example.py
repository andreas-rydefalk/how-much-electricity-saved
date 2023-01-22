import matplotlib.pyplot as plt
import pandas as pd

from how_much_electricity_saved import fitting_functions
from how_much_electricity_saved.date_range import DateRange
from how_much_electricity_saved.main import ElectricityConsumptionReport


def main():

    df = pd.read_csv("examples/sample_daily_consumption.csv")

    # Convert the time column to datetime
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    fitting_func = fitting_functions.fitting_func_exp

    report = ElectricityConsumptionReport(
        dataframe=df,
        baseline_period=DateRange(begin="2022-01-01", end="2022-09-10"),
        fitting_function=fitting_func,
        initial_guess=(1, -1, 1),
        dateranges_to_group_to=[
            DateRange(begin="2022-01-01", end="2022-09-10", color="blue"),
            DateRange(begin="2022-09-10", end="2022-12-22", color="orange"),
        ],
        dot_size=10,
    )

    fig = plt.figure(figsize=(18, 18))
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(313)

    report.create_scatter_plot_total(ax=ax1)
    report.create_plot_cumnulative_consumed_electricity(ax=ax2)
    report.create_plot_cumnulative_saved_electricity_vs_time(ax=ax3)
    report.create_plot_cumnulative_saved_electricity_percents(ax=ax4)
    report.create_plot_monthly_bar_chart(ax=ax5)

    fig.savefig("examples/example_output.png")


if __name__ == "__main__":
    main()
