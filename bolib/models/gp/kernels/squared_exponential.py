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

import bolib.models.gp.kernels.util


def kernel_function(mat_a, mat_b, lengthscale):
    """Squared Exponential covariance function
    $k_{SE} \left(r\right) = exp \left( - \dfrac{r^2}{2l^2} \right) $"""
    sq_dist = bolib.models.gp.kernels.util.sq_distance(
        mat_a, mat_b, lengthscale)
    return stationary_function(sq_dist)


def dk_dx(mat_a, mat_b, lengthscale):
    """ gradient of the kernel function """
    dr_dx = bolib.models.gp.kernels.util.dr_dx(
        mat_a, mat_b, lengthscale)
    sq_dist = np.array(bolib.models.gp.kernels.util.sq_distance(
        mat_a, mat_b, lengthscale))
    return -0.5 * np.exp(-0.5*sq_dist)[:, :, np.newaxis] * dr_dx


def dk_dl(mat_a, mat_b, lengthscale):
    """ gradient of the kernel function """
    dr_dl = bolib.models.gp.kernels.util.dr_dl(
        mat_a, mat_b, lengthscale)
    sq_dist = np.array(bolib.models.gp.kernels.util.sq_distance(
        mat_a, mat_b, lengthscale))
    return -0.5 * np.exp(-0.5*sq_dist)[:, :, np.newaxis] * dr_dl


def stationary_function(sq_dist):
    """ stationary function """
    return np.matrix(np.exp(-0.5*sq_dist))
