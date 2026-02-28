import os
import sys
import atexit
import logging
import traceback
from time import time

from utils.config import Config
from utils.logger.time_formatter import TimeFormatter

try:
	# POSIX file lock
	import fcntl
	_HAS_FCNTL = True
except Exception:
	_HAS_FCNTL = False


class Logger:
	def __init__(self, name: str):
		self.name = name
		self.cfg = Config().logger

		# Create the Path if doesnt exist
		os.makedirs(self.cfg.path, exist_ok=True)

		# Current File Path
		self.f_path = os.path.join(self.cfg.path, f"{self.name}.log")

		# File
		self._fd = None
		# Closed Flag
		self._closed = False

		# Time Formatter
		self._time_fmt = TimeFormatter()

		# In-memory batching buffer
		self._buf = bytearray()
		atexit.register(self._atexit_flush_close)

		# Log Level
			# Default is INFO
		self._LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR,}
		self.level = self._LEVELS.get(self.cfg.level, logging.INFO)

	def debug(self, msg: str):
		self._log(logging.DEBUG, msg)

	def info(self, msg: str):
		self._log(logging.INFO, msg)

	def warning(self, msg: str):
		self._log(logging.WARNING, msg)

	def error(self, msg: str):
		self._log(logging.ERROR, msg)

	def _log(self, level: int, message: str):
		# Skip logging for stuff lower than the current level
		if self._closed or level < self.level:
			return

		t = time()
		# Format Time
		ts = self._time_fmt.format(t)
		level_name = logging.getLevelName(level)
		# Build once with minimal allocations
		line = f"{ts} - {level_name} - {self.name} - {message}\n"

		# Console (cheap path: no per-call flush)
		if self.cfg.console:
			try:
				sys.stdout.write(line)
				if self.cfg.console_flush:
					sys.stdout.flush()
			except Exception:
				# Avoid raising from logger
				try:
					sys.stderr.write("Logger console write error\n")
				except Exception:
					pass

		# File
		data = line.encode("utf-8", "strict")
		if self.cfg.buf_threshold > 0:
			# Extend the Byte array
			self._buf.extend(data)
			# Write to file if the current bytearray length is bigger than the threshold
			if len(self._buf) >= self.cfg.buf_threshold:
				self._write_bytes(self._buf)
				self._buf.clear()
		else:
			self._write_bytes(data)

	def _write_bytes(self, data: bytes | bytearray) -> None:
		# Skip if empty
		if not data or self._closed:
			return

		# Lock
		if self.cfg.lock and _HAS_FCNTL:
			try:
				fcntl.flock(self.fd, fcntl.LOCK_EX)
			except Exception:
				pass
		try:
			mv = memoryview(data)
			total = 0
			while total < len(mv):
				n = os.write(self.fd, mv[total:])
				if n <= 0:
					break
				total += n
		except Exception:
			try:
				sys.stderr.write("Error writing log to file\n")
			except Exception:
				traceback.print_exc()
		finally:
			if self.cfg.lock and _HAS_FCNTL:
				try:
					fcntl.flock(self.fd, fcntl.LOCK_UN)
				except Exception:
					pass

	def set_level(self, level_name_or_int):
		if isinstance(level_name_or_int, int):
			self.level = level_name_or_int
		else:
			self.level = self._LEVELS.get(str(level_name_or_int).lower(), self.level)

	def shutdown(self):
		# Flush and close resources (synchronous)
		if self._closed:
			return
		try:
			self.flush()
		finally:
			try:
				os.close(self._fd)
			except Exception:
				pass
			self._closed = True

	def flush(self):
		# Force-flush any buffered log lines to disk
		if self._closed:
			return
		if self._buf:
			self._write_bytes(self._buf)
			self._buf.clear()
		# stdout flush is optional
		if self.cfg.console and self.cfg.console_flush:
			try:
				sys.stdout.flush()
			except Exception:
				pass

	def _atexit_flush_close(self):
		# Best-effort finalization without exceptions bubbling up
		try:
			self.shutdown()
		except Exception:
			pass

	@property
	def fd(self):
		if self._fd is None:
			# Open file once
			# O_APPEND ensures correct append semantics even across processes
			flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
			# Windows
			if hasattr(os, "O_BINARY"):
				flags |= os.O_BINARY
			self._fd = os.open(self.f_path, flags, 0o644)
			return self._fd
		else:
			return self._fd

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		self.shutdown()