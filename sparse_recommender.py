#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:56:24 2023

@author: tsi2
"""

from scipy import sparse
import scipy.sparse as sps
from scipy.sparse import random
from scipy.sparse import coo_matrix
import numpy as np

class SparseMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = []   # List
        self.row_indices = []  
        self.col_indices = []  

    def add_value(self, row, col, value):#Sets the value at (row, col) to value
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.data.append(value)
            self.row_indices.append(row)
            self.col_indices.append(col)
        else:
            raise ValueError("Invalid row or column index")

    def get_value(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            index = self.row_indices.index(row) if row in self.row_indices else -1
            if index != -1 and self.col_indices[index] == col:
                return self.data[index]
            else:
                return 0  
        else:
            raise ValueError("Invalid row or column index")

    def to_dense(self): #Converts the sparse matrix to a dense matrix and return it.
        dense_matrix = [[0] * self.cols for _ in range(self.rows)]
        for i in range(len(self.data)):
            dense_matrix[self.row_indices[i]][self.col_indices[i]] = self.data[i]
        return dense_matrix

    def get_shape(self): #Returns the value at (row, col).
        return self.rows, self.cols
        
    def matrix_vector_multiply(self, user_vector):
        if len(user_vector) != self.cols:
            raise ValueError("User vector length must match the number of columns in the matrix")

        result = [0] * self.rows

        for i in range(len(self.data)):
            row = self.row_indices[i]
            col = self.col_indices[i]
            value = self.data[i]
            result[row] += value * user_vector[col]

        return result #return the result.
        
    def matrix_addition(self, other_matrix):
        if self.get_shape() != other_matrix.get_shape():
            raise ValueError("Matrix dimensions must match for addition")

        result_matrix = SparseMatrix(self.rows, self.cols)

        for i in range(len(self.data)):
            result_matrix.add_element(
                self.row_indices[i], self.col_indices[i], self.data[i]
            )

        for i in range(len(other_matrix.data)):
            result_matrix.add_element(
                other_matrix.row_indices[i],
                other_matrix.col_indices[i],
                other_matrix.data[i],
            )

        return result_matrix #return the result.







