#
# Copyright 2023 Martin Ladecky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
The entry point for the code
"""

__version__ = '0.0.1'
# old verbose
# _labels = {
#     'true_error': r'$\overline{\varepsilon}^{T} (A_{h,k}^{\mathrm{eff}} -A^{\mathrm{eff}}_{h,\infty})\,\overline{\varepsilon} $ ',
#     'trivial_upper_bound': r'Trivial upper bound - $\frac{1}{\lambda_{min}}|| \mathbf{ r}_{k} ||_{\mathbf{G}}^2$',
#     'trivial_lower_bound': r'Trivial lower bound - $\frac{1}{\lambda_{max}}|| \mathbf{ r}_{k} ||_{\mathbf{G}}^2$',
#     'PT_upper_bound': 'Upper bound PT',
#     'PT_lower_bound': 'Lower bound PT',
#     'PT_upper_estimate': 'Upper estimate PT',
# }
# new short
_labels = {
    'true_error': r'Iterative error - $ e_{h,k}  $ ',
    'trivial_upper_bound': r'Trivial upper bound - $\frac{1}{\lambda_{min}}|| \mathbf{ r}_{k} ||_{\mathbf{G}}^2$',
    'trivial_lower_bound': r'Trivial lower bound - $\frac{1}{\lambda_{max}}|| \mathbf{ r}_{k} ||_{\mathbf{G}}^2$',
    'PT_upper_bound': 'Upper bound PT',
    'PT_lower_bound': 'Lower bound PT',
    'PT_upper_estimate': 'Upper estimate PT',
}

_colors = {
    'true_error': 'black',
    'trivial_upper_bound': r'blue',
    'trivial_lower_bound': r'blue',
    'PT_upper_bound': r'green',
    'PT_lower_bound': r'red',
    'PT_upper_estimate': r'red',
    '0': 'black',
    '1': 'red',
    '2': 'blue',
    '3': 'green',
    '4': 'orange',
    '5': 'purple'
}

_markers = {
    'true_error': 'x',
    'trivial_upper_bound': 'v',
    'trivial_lower_bound': '^',
    'PT_upper_bound': r'v',
    'PT_lower_bound': r'^',
    'PT_upper_estimate': r'v',
    '0': '|',
    '1': 'x',
    '2': 'o',
    '3': '^',
    '4': 'v',
    '5': '.'
}
_linestyles = {
    '0': '-',
    '1': ':',
    '2': '-.',
    '3': '--',
}

# linestyles = [':', '-.', '--', (0, (3, 1, 1, 1))]
# colors = ['red', 'blue', 'green', 'orange', 'purple','olive','brown','purple']
#
# # markers = ['x', 'o', '|', '>']
# markers = ["x",  "|", ".","v", "<", ">","o",  "^", ".", ",", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+",
#            "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
#            ]
