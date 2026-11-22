"""Shared pytest configuration.

Some tests draw diagnostic plots and call ``plt.show()``. With an interactive
backend this opens a window and blocks the test run (and the pre-commit hook)
until the window is closed. Selecting the non-interactive ``Agg`` backend
before any test imports ``matplotlib.pyplot`` makes ``plt.show()`` a no-op.
"""
import matplotlib

matplotlib.use('Agg')
