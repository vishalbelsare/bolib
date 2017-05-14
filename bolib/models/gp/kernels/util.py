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
    """ Measures the distance matrix between solutions of A and B.
    
    :param mat_a: List of solutions in lines and dimensions in columns.
    :type mat_a: np.matrix 
    :param mat_b: List of solutions in lines and dimensions in columns.
    :type mat_b: np.matrix
    :param lengthscale: Array of lenghtscale parameters. One per dimension
     in ARD case, only one element otherwise.  
    :type lengthscale: np.array
    :return: Distance matrix between solutions of A and B.
    :rtype: np.matrix """
    mat_a = mat_a / lengthscale
    mat_b = mat_b / lengthscale
    result = np.sum(np.power(mat_a, 2.0), axis=1) + \
        np.sum(np.power(mat_b, 2.0), axis=1).T - 2 * np.dot(mat_a, mat_b.T)

    return result.clip(min=0.0)


def dr_dx(mat_a, mat_b, lengthscale):
    """ 
    Measures gradient of the distance between solutions of A and B in X.
    
    :param mat_a: List of solutions in lines and dimensions in columns.
    :type mat_a: np.matrix 
    :param mat_b: List of solutions in lines and dimensions in columns.
    :type mat_b: np.matrix
    :param lengthscale: Array of lenghtscale parameters. One per dimension
     in ARD case, only one element otherwise.  
    :type lengthscale: np.array
    :return: 3D array with the gradient in every dimension of X.
    :rtype: np.array """
    mat_a = mat_a
    mat_b = mat_b

    result = np.zeros((mat_a.shape[0], mat_b.shape[0], mat_a.shape[1]))

    for i in range(0, mat_a.shape[0]):
        result[i, :, :] = 2.0 * (mat_a[i, :] - mat_b[:, :]) /\
            np.power(lengthscale, 2.0)

    return result


def dr_dl(mat_a, mat_b, lengthscale):
    """ 
    Measures gradient of the distance between solutions of A and B in the
    length-scale hyper-parameter space.

    :param mat_a: List of solutions in lines and dimensions in columns.
    :type mat_a: np.matrix 
    :param mat_b: List of solutions in lines and dimensions in columns.
    :type mat_b: np.matrix
    :param lengthscale: Array of lenghtscale parameters. One per dimension
     in ARD case, only one element otherwise.  
    :type lengthscale: np.array
    :return: 3D array with the gradient in every
     dimension the length-scale hyper-parameter space.
    :rtype: np.array """
    mat_a = mat_a
    mat_b = mat_b

    result = np.zeros((mat_a.shape[0], mat_b.shape[0], mat_a.shape[1]))

    for i in range(0, mat_a.shape[0]):
        result[i, :, :] = - 2.0 * (np.power(mat_a[i, :] - mat_b[:, :], 2.0)) /\
            np.power(lengthscale, 3.0)

    return result
