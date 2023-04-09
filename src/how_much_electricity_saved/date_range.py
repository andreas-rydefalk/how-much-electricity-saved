import pandas as pd


class DateRange:
    def __init__(self, begin: str, end: str, color: str = "blue"):
        """Input begin and end dates in a format that pandas is able parse to date.
        Such as the ISO format YYYY-MM-DD"""

        self.begin = pd.to_datetime(begin).date()
        self.end = pd.to_datetime(end).date()
        self.color = color

    def timestamp_in_date_range(self, timestamp):
        return timestamp.date() >= self.begin and timestamp.date() < self.end

    def __repr__(self) -> str:
        return f'DateRange("{self.begin}", "{self.end}")'
