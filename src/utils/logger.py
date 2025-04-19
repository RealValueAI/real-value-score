import logging
from typing import Optional, Union

import pandas as pd


class Logger:
    """Just logger."""

    def __init__(self, name: str = 'MLLogger', level: int = logging.INFO):
        """Initialise the logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - '
            '%(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        self.logger.addHandler(ch)

        self.log_messages = {
            "start_fetch": "Start fetching data from Oracle",
        }

    def log_df_info(self, df: pd.DataFrame,
                    message: Optional[str] = None) -> None:
        """Log the shape of a pandas DataFrame."""
        if message:
            self.logger.info(f'{message}, shape {df.shape}')
        else:
            self.logger.info(f' shape {df.shape}')

    def log_check_nulls(self, df: pd.DataFrame,
                        message: Optional[str] = None) -> None:
        """Log the number of null values in a pandas DataFrame."""
        if message:
            nulls = df.isna().sum()[df.isna().sum() > 0].to_dict()
            self.logger.info(f'{message}, nulls: {nulls}')
        else:
            self.logger.info(f'shape {df.shape}')

    def log_check_duplicates(
            self,
            df: pd.DataFrame, message: Union[str, None] = None,
    ) -> None:
        """Log the number of duplicate rows in a pandas DataFrame."""
        if message:
            self.logger.info(f'{message} {df.duplicated(keep="first").sum()}')
        else:
            self.logger.info(
                f'Duplicates: {df.duplicated(keep="first").sum()}')

    def log_message(
            self,
            message: Optional[str],
            level: int = logging.INFO,
    ) -> None:
        """Log a message."""
        if message:
            self.logger.log(level, message)
        else:
            self.logger.info('logging')

    def info(self, message: Optional[str] = None) -> None:
        """Log simple info."""
        self.logger.info(message)

    def log_predefined_message(self, key: str) -> None:
        """Log a predefined message based on a key."""
        if key in self.log_messages:
            self.log_message(self.log_messages[key])
        else:
            self.logger.warning(f'No predefined message found for key: {key}')


logger = Logger()