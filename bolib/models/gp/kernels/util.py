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


def sq_distance(mat_a, mat_b, lengthscale):
    """ Square Disance """
    mat_a = mat_a / lengthscale
    mat_b = mat_b / lengthscale
    result = np.sum(np.power(mat_a, 2.0), axis=1) + \
        np.sum(np.power(mat_b, 2.0), axis=1).T - 2 * np.dot(mat_a, mat_b.T)

    return result.clip(min=0.0)


def dr_dx(mat_a, mat_b, lengthscale):
    """ Gradient of the square Disance """
    mat_a = mat_a
    mat_b = mat_b

    result = np.zeros((mat_a.shape[0], mat_b.shape[0], mat_a.shape[1]))

    for i in range(0, mat_a.shape[0]):
        result[i, :, :] = 2.0 * (mat_a[i, :] - mat_b[:, :]) /\
            np.power(lengthscale, 2.0)

    return result


def dr_dl(mat_a, mat_b, lengthscale):
    """ Gradient of the square Disance """
    mat_a = mat_a
    mat_b = mat_b

    result = np.zeros((mat_a.shape[0], mat_b.shape[0], mat_a.shape[1]))

    for i in range(0, mat_a.shape[0]):
        result[i, :, :] = - 2.0 * (np.power(mat_a[i, :] - mat_b[:, :], 2.0)) /\
            np.power(lengthscale, 3.0)

    return result
