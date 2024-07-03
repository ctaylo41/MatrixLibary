#include <vector>
#include "matrix.h"
#include <iostream>
#include <string>

Matrix::Matrix(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;
    this->data.resize(rows);
    for (int i = 0; i < rows; i++)
    {
        data[i].resize(cols);
    }
}

Matrix::Matrix(int n)
{
    this->rows = n;
    this->cols = n;
    this->data.resize(n);
    for (int i = 0; i < n; i++)
    {
        data[i].resize(n, 0);
        data[i][i] = 1;
    }
}

Matrix::Matrix(std::vector<std::vector<std::complex<double> > > data)
{
    this->data = data;
    for (int i = 0; i < data.size(); i++)
    {
        if (data[i].size() != data[0].size())
        {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
    this->rows = data.size();
    this->cols = data[0].size();
}

std::vector<std::vector<std::complex<double> > > Matrix::getData()
{
    return data;
}

std::complex<double> Matrix::get(int i, int j)
{
    return data[i][j];
}

int Matrix::getRows()
{
    return rows;
}

int Matrix::getCols()
{
    return cols;
}

void Matrix::set(int i, int j, double value)
{
    data[i][j] = value;
}

Matrix Matrix::transpose()
{
    Matrix a = *this;
    Matrix result(a.cols, a.rows);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            result.data[j][i] = a.data[i][j];
        }
    }

    return result;
}
Matrix Matrix::inverse()
{
    Matrix a = *this;
    if (a.rows != a.cols)
    {
        throw std::invalid_argument("Matrix is not square");
    }
    int n = a.rows;
    Matrix result(n);
    Matrix L(n);
    Matrix U(n);
    Matrix P(n);    
    LUDecomposition(a, L, U, P);
    
    Matrix b(n);
    Matrix solve = P*b;
    Matrix y = L.forwardSolve(solve);
    Matrix x = U.backwardSolve(y);
    
    return x;
    
}


Matrix Matrix::choleskyDecomp()
{
    Matrix a = *this;
    if (a.rows != a.cols)
    {
        throw std::invalid_argument("Matrix is not square");
    }
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            std::complex<double> sum = 0;
            for (int k = 0; k < j; k++)
            {
                sum += result.data[i][k] * result.data[j][k];
            }
            if (i == j)
            {
                std::complex<double> value = a.data[i][i] - sum;
                if (value.real() < 0)
                {
                    throw std::runtime_error("Matrix is not positive definite");
                }
                result.data[i][j] = sqrt(value);
            }
            else
            {
                if (result.data[j][j] == std::complex<double>(0, 0))
                {
                    throw std::runtime_error("Division by zero encountered in Cholesky decomposition");
                }
                result.data[i][j] = (a.data[i][j] - sum) / result.data[j][j];
            }
        }
    }
    return result;
}

bool Matrix::equals(Matrix &a, Matrix &b)
{
    if (a.rows != b.rows || a.cols != b.cols)
    {
        return false;
    }
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            if (std::abs(a.data[i][j]-b.data[i][j])>1e-9)
            {
                return false;
            }
        }
    }
    return true;
}

Matrix Matrix::backwardSolve(Matrix &b)
{
    Matrix a = *this;
    if (a.rows != a.cols)
    {
        throw std::invalid_argument("Matrix is not square");
    }
    if (a.rows != b.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    Matrix result(a.rows, b.cols);
    for (int i = a.rows - 1; i >= 0; i--)
    {
        for (int j = 0; j < b.cols; j++)
        {
            std::complex<double> sum = 0;
            for (int k = i + 1; k < a.cols; k++)
            {
                sum += a.data[i][k] * result.data[k][j];
            }
            result.data[i][j] = (b.data[i][j] - sum) / a.data[i][i];
        }
    }
    return result;
}
Matrix Matrix::forwardSolve(Matrix &b)
{
    Matrix a = *this;
    if (a.rows != a.cols)
    {
        throw std::invalid_argument("Matrix is not square");
    }
    if (a.rows != b.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    Matrix result(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < b.cols; j++)
        {
            std::complex<double> sum = 0;
            for (int k = 0; k < i; k++)
            {
                sum += a.data[i][k] * result.data[k][j];
            }
            result.data[i][j] = (b.data[i][j] - sum) / a.data[i][i];
        }
    }
    return result;
}

void Matrix::LUDecomposition(Matrix &a, Matrix &L, Matrix &U, Matrix &P)
{
    if (a.rows != a.cols)
    {
        throw std::invalid_argument("Matrix is not square");
    }

    int n = a.rows;

    L = Matrix(n);
    U = Matrix(n);
    P = Matrix(n);

    for (int k = 0; k < n - 1; ++k) {
        double maxval = 0.0;
        int maxindex = k;
        // Find the maximum value and its index
        for (int i = k; i < n; ++i) {
            if (std::abs(a.get(i,k)) > maxval) {
                maxval = std::abs(a.get(i,k));
                maxindex = i;
            }
        }
        int q = maxindex;
        if (maxval == 0) throw std::runtime_error("A is singular");
        if (q != k) {
            // Swap rows in A and P
            std::swap(a.data[k], a.data[q]);
            std::swap(P.data[k], P.data[q]);
        }
        // Update the elements below the pivot
        for (int i = k + 1; i < n; ++i) {
            a.data[i][k] /= a.data[k][k];
            for (int j = k + 1; j < n; ++j) {
                a.data[i][j] -= a.data[i][k] * a.data[k][j];
            }
        }
    }

    // Split the LU matrix into L and U
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > j) {
                L.data[i][j] = a.data[i][j];
                U.data[i][j] = 0;
            } else if (i == j) {
                L.data[i][j] = 1;
                U.data[i][j] = a.data[i][j];
            } else {
                L.data[i][j] = 0;
                U.data[i][j] = a.data[i][j];
            }
        }
    }
}

std::string Matrix::toString()
{
    std::string result = "";
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result += std::to_string(data[i][j].real()) + " " + std::to_string(data[i][j].imag()) + "i ";
        }
        result += "\n";
    }
    return result;
}

std::complex<double> Matrix::determinant()
{   Matrix a = *this;
    if (a.rows != a.cols)
    {
        throw std::invalid_argument("Matrix is not square");
    }
    int n = a.rows;
    std::complex<double> det = 1;
    Matrix L(n);
    Matrix U(n);
    Matrix P(n);
    LUDecomposition(a, L, U, P);
    for (int i = 0; i < n; i++)
    {
        det *= U.data[i][i];
    }
    return det;
}

Matrix Matrix::operator*(Matrix &b)
{
    Matrix a = *this;
    if (a.cols != b.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    Matrix result(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < b.cols; j++)
        {
            for (int k = 0; k < a.cols; k++)
            {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

Matrix Matrix::operator+(Matrix &b)
{
    Matrix a = *this;
    if (a.rows != b.rows || a.cols != b.cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            result.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(Matrix &b)
{
    Matrix a = *this;
    if (a.rows != b.rows || a.cols != b.cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            result.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(double b)
{
    Matrix a = *this;
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            result.data[i][j] = a.data[i][j] * b;
        }
    }
    return result;
}


Matrix Matrix::operator/(Matrix &b)
{
    Matrix a = *this;
    Matrix L(a.rows);
    Matrix U(a.rows);
    Matrix P(a.rows);
    LUDecomposition(a, L, U, P);
    Matrix result = P*b;
    Matrix y = L.forwardSolve(result);
    Matrix x = U.backwardSolve(y);
    return x;
}

bool Matrix::isSquare()
{
    return rows == cols;
}

bool Matrix::isUpperHessenberg() {
    Matrix a = *this;
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            if (i - j > 1 && a.data[i][j].real() != 0) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::isTriDiagonal() {
    Matrix a = *this;
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            if (std::abs(i - j) > 1 && a.data[i][j].real() != 0) {
                return false;
            }
        }
    }
    return true;
}

Matrix Matrix::QRSolver(Matrix &b) {
    
}