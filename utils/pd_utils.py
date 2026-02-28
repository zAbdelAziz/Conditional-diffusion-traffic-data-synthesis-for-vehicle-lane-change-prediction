from typing import Union
from pandas import DataFrame, to_numeric


def de_to_numeric(df: DataFrame, columns: Union[list, tuple] = None):
	if columns is None:
		columns = df.columns

	for column in columns:
		s = df[column].astype(str)
		# remove quotes, strip whitespace
		s = s.str.replace('"', "", regex=False).str.strip()
		# decimal comma to decimal dot
		s = s.str.replace(",", ".", regex=False)
		df[column] = to_numeric(s, errors="coerce")
	return df