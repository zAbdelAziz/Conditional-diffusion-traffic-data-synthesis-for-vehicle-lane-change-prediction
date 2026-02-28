from time import strftime, localtime


class TimeFormatter:
	__slots__ = ("last_sec", "cached_prefix")

	def __init__(self):
		self.last_sec = -1
		self.cached_prefix = ""

	def format(self, t: float) -> str:
		sec = int(t)
		if sec != self.last_sec:
			# Recompute only when second changes
			self.last_sec = sec
			self.cached_prefix = strftime("%Y-%m-%d %H:%M:%S", localtime(sec))
		ms = int((t - sec) * 1000.0)
		# YYYY-mm-dd HH:MM:SS.mmm
		return f"{self.cached_prefix}.{ms:03d}"