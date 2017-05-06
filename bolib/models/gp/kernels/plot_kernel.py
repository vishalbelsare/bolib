# -*- coding: utf-8 -*-
#
#    Copyright 2017 Ibai Roman
#
#    This file is part of BOlib.
#
#    BOlib is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    BOlib is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with BOlib. If not, see <http://www.gnu.org/licenses/>.
import numpy as np
from matplotlib import pyplot

import bolib.loader


def plot_kernel_implementation(k):
    """ Plot a kernel implementation """
    assert 'stationary_function' in dir(k), "This is not a stationary kernel"
    pyplot.figure()
    pyplot.clf()
    dist = np.linspace(0, 1, 100)
    kern = np.asarray(
        k.stationary_function(np.power(np.matrix(dist), 2))).reshape(-1)
    pyplot.plot(dist, kern, 'b-', ms=40)
    pyplot.title(k.__name__)
    pyplot.xlabel("input distance, r")
    pyplot.ylabel("covariance, k(r)")
    pyplot.show()

if __name__ == "__main__":
    kernel = bolib.loader.load_entry_point(
        'bolib.models.gp.kernels', 'matern52')
    plot_kernel_implementation(kernel)
