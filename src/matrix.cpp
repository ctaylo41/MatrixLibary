#include <vector>
#include "matrix.h"
#include <iostream>
#include <string>
#include "vector.h"
#include <cmath>

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
        for (int j = 0; j < data[i].size(); j++)
        {
            if (data[i][j].imag() != 0)
            {
                this->complex = true;
            }
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

void Matrix::set(int i, int j, std::complex<double> value)
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
    Matrix solve = P * b;
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
            if (std::abs(a.data[i][j] - b.data[i][j]) > 1e-9)
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
    if (a.rows > a.cols)
    {
        throw std::invalid_argument("Matrix has more rows than columns, not supported for this operation.");
    }
    if (a.rows != b.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    Matrix result(a.cols, b.cols);
    for (int i = a.rows - 1; i >= 0; i--)
    {
        for (int j = 0; j < b.cols; j++)
        {
            std::complex<double> sum = 0;
            for (int k = i + 1; k < a.cols; k++)
            {
                sum += a.data[i][k] * result.data[k][j];
            }
            if (std::abs(a.data[i][i]) < 1e-12) // Check for division by zero or very small numbers
            {
                throw std::runtime_error("Matrix is singular or nearly singular");
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

    for (int k = 0; k < n - 1; ++k)
    {
        double maxval = 0.0;
        int maxindex = k;
        // Find the maximum value and its index
        for (int i = k; i < n; ++i)
        {
            if (std::abs(a.get(i, k)) > maxval)
            {
                maxval = std::abs(a.get(i, k));
                maxindex = i;
            }
        }
        int q = maxindex;
        if (maxval == 0)
            throw std::runtime_error("A is singular");
        if (q != k)
        {
            // Swap rows in A and P
            std::swap(a.data[k], a.data[q]);
            std::swap(P.data[k], P.data[q]);
        }
        // Update the elements below the pivot
        for (int i = k + 1; i < n; ++i)
        {
            a.data[i][k] /= a.data[k][k];
            for (int j = k + 1; j < n; ++j)
            {
                a.data[i][j] -= a.data[i][k] * a.data[k][j];
            }
        }
    }

    // Split the LU matrix into L and U
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i > j)
            {
                L.data[i][j] = a.data[i][j];
                U.data[i][j] = 0;
            }
            else if (i == j)
            {
                L.data[i][j] = 1;
                U.data[i][j] = a.data[i][j];
            }
            else
            {
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
{
    Matrix a = *this;
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
    if (a.isSquare())
    {
        if (a.isUpperTriangular())
        {
            return a.backwardSolve(b);
        }
        else if (a.isLowerTriangular())
        {
            return a.forwardSolve(b);
        }
        else if (a.complex && a.getCols() <= 16 || !a.complex && a.getCols() <= 8)
        {
            return a.LUSolver(b);
        }
        else
        {
            if (a.isUpperHessenberg())
            {
                if (a.isTriDiagonal())
                {
                    return a.tridigonalSolver(b);
                }
                else
                {
                    return a.hessenbergSolver(b);
                }
            }
            else if (a.isHermitian())
            {
                try
                {
                    a = a.choleskyDecomp();
                    return a.backwardSolve(b);
                }
                catch (std::runtime_error e)
                {
                    return a.LUSolver(b);
                }
            }
            else
            {
                return a.LUSolver(b);
            }
        }
    }
    else
    {
        return a.QRSolver(b);
    }
}

bool Matrix::isSquare()
{
    return rows == cols;
}

bool Matrix::isUpperHessenberg()
{
    Matrix a = *this;
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            if (i - j > 1 && a.data[i][j].real() != 0)
            {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::isTriDiagonal()
{
    Matrix a = *this;
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            if (std::abs(i - j) > 1 && a.data[i][j].real() != 0)
            {
                return false;
            }
        }
    }
    return true;
}

Matrix Matrix::GramSchmidt()
{
    Matrix a = *this;

    a = a.transpose();
    Matrix res(a.rows, a.cols);
    for (int i = 0; i < a.cols; i++)
    {
        Vector col = Vector(a.getRow(i).getData()[0]);
        for (int j = 0; j < i; j++)
        {
            Vector proj = Vector(res.data[j]);
            std::complex<double> dot = col * proj;
            Vector temp = proj * dot;
            col = col - temp;
        }

        res.data[i] = col.norm().transpose().getData()[0];
    }
    return res.transpose();
}

Matrix Matrix::getRow(int i)
{
    Matrix a = *this;
    Matrix result(1, a.cols);
    for (int j = 0; j < a.cols; j++)
    {
        result.data[0][j] = a.data[i][j];
    }
    return result;
}

Matrix Matrix::getCol(int i)
{
    Matrix a = *this;
    Matrix result(a.rows, 1);
    for (int j = 0; j < a.rows; j++)
    {
        result.data[j][0] = a.data[j][i];
    }
    return result;
}

void Matrix::QRDecomposition(Matrix &A, Matrix &Q, Matrix &R)
{
    int m = A.cols;
    int n = A.rows;

    Q = Matrix(n, n);
    R = Matrix(n, m);
    for (int i = 0; i < n; i++)
    {
        Vector col = Vector(A.getCol(i).transpose().getData()[0]);
        for (int j = 0; j < i; j++)
        {
            Vector proj = Vector(Q.data[j]);
            std::complex<double> dot = col * proj;
            R.data[j][i] = dot;
            Vector temp = proj * dot;

            col = col - temp;
        }

        R.data[i][i] = col.magnitude();
        Vector tmp = col.norm();
        Q.data[i] = tmp.transpose().getData()[0];
    }
    if (n < m)
    {
        for (int i = n; i < m; i++)
        {
            Vector col = Vector(A.getCol(i).transpose().getData()[0]);
            for (int j = 0; j < n; j++)
            {
                Vector proj = Vector(Q.data[j]);
                std::complex<double> dot = col * proj;
                R.data[j][i] = dot; // Update R with the dot product
            }
        }
    }

    Q = Q.transpose();
}

Matrix Matrix::conjugate()
{
    Matrix a = *this;
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            result.data[i][j] = std::conj(a.data[i][j]);
        }
    }

    return result;
}

bool Matrix::isHermitian()
{
    Matrix a = *this;
    if (a.rows != a.cols)
    {
        return false;
    }
    Matrix conj = a.conjugate().transpose();
    return equals(a, conj);
}

Matrix Matrix::QRSolver(Matrix &b)
{
    Matrix a = *this;
    Matrix Q(a.rows, a.cols);
    Matrix R(a.rows, a.cols);
    Matrix::QRDecomposition(a, Q, R);

    Matrix y = Q.transpose() * b;
    Matrix x = R.backwardSolve(y);
    return x;
}
bool Matrix::isUpperTriangular()
{
    Matrix a = *this;
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < i; j++)
        {
            if (a.data[i][j].real() != 0)
            {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::isLowerTriangular()
{
    Matrix a = *this;
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = i + 1; j < a.cols; j++)
        {
            if (a.data[i][j].real() != 0)
            {
                return false;
            }
        }
    }
    return true;
}

Matrix Matrix::tridigonalSolver(Matrix &b)
{
    Matrix a = *this;
    Matrix bCopy = b;
    if (!a.isTriDiagonal())
    {
        throw std::invalid_argument("Matrix is not tridiagonal");
    }
    if (a.rows != b.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    Matrix result(a.rows, b.cols);
    std::vector<std::complex<double> > c(a.rows);
    std::vector<std::complex<double> > d(a.rows);
    std::vector<std::complex<double> > e(a.rows);
    for (int i = 0; i < a.rows; i++)
    {
        if (i < a.rows - 1)
        {
            c[i] = this->data[i][i + 1]; // Superdiagonal
            e[i] = this->data[i + 1][i]; // Subdiagonal
        }
        d[i] = this->data[i][i];
    }
    for (int i = 1; i < a.rows; i++)
    {
        std::complex<double> temp = e[i - 1] / d[i - 1];
        d[i] = d[i] - temp * c[i - 1];
        bCopy.data[i][0] = bCopy.data[i][0] - temp * bCopy.data[i - 1][0];
    }

    for (int i = a.rows - 1; i >= 0; i--)
    {
        bCopy.data[i][0] = bCopy.data[i][0] / d[i];
        if (i > 0)
        {
            bCopy.data[i - 1][0] = bCopy.data[i - 1][0] - c[i - 1] * bCopy.data[i][0];
        }
    }

    return bCopy;
}

Matrix Matrix::calculateGivenRotationMatrix(int i, int j)
{
    Matrix H = *this;
    if (H.getRows() != H.getCols())
    {
        throw std::invalid_argument("Matrix is not square");
    }

    std::complex<double> a = H.get(i, j);
    std::complex<double> b = H.get(i + 1, j);
    std::complex<double> r = std::sqrt(a * a + b * b);
    std::complex<double> c = a / r;
    std::complex<double> s = -b / r;

    Matrix res = Matrix(H.getRows());
    res.data[i][j] = c;
    res.data[i][j + 1] = -s;
    res.data[i + 1][j] = s;
    res.data[i + 1][j + 1] = c;
    return res;
}

void Matrix::hessenbergQRDecomposition(Matrix &A, Matrix &Q, Matrix &R)
{
    if (A.getRows() != A.getCols())
    {
        throw std::invalid_argument("Matrix is not square");
    }
    R = A;
    Q = Matrix(A.getRows());
    for (int i = 0; i < A.getRows() - 1; i++)
    {
        Matrix G = R.calculateGivenRotationMatrix(i, i);
        R = G * R;
        Matrix tmp = G.transpose();
        Q = Q * tmp;
    }
    Matrix s = Q * R;
}

void Matrix::setRow(int i, Matrix &row)
{
    Matrix a = *this;
    if (a.cols != row.cols)
    {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    for (int j = 0; j < a.cols; j++)
    {
        if (row.data[i][j].imag() != 0)
        {
            this->complex = true;
        }
        a.data[i][j] = row.data[0][j];
    }
}

void Matrix::setCol(int i, Matrix &col)
{
    Matrix a = *this;
    if (a.rows != col.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    for (int j = 0; j < a.rows; j++)
    {
        a.data[j][i] = col.data[j][0];
        if (col.data[j][0].imag() != 0)
        {
            this->complex = true;
        }
    }
}

Matrix Matrix::hessenbergSolver(Matrix &b)
{
    Matrix a = *this;
    Matrix Q(a.getRows());
    Matrix R(a.getRows());
    Matrix::hessenbergQRDecomposition(a, Q, R);
    Matrix y = Q.transpose() * b;
    Matrix x = R.backwardSolve(y);
    return x;
}

Matrix Matrix::LUSolver(Matrix &b)
{
    Matrix a = *this;
    Matrix L(a.rows);
    Matrix U(a.rows);
    Matrix P(a.rows);
    LUDecomposition(a, L, U, P);
    Matrix result = P * b;
    Matrix y = L.forwardSolve(result);
    Matrix x = U.backwardSolve(y);
    return x;
}

bool Matrix::isSparse()
{
    Matrix a = *this;
    int count = 0;
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            if (a.data[i][j].real() != 0)
            {

                count++;
            }
        }
    }
    float res = static_cast<float>(count) / (a.getRows() * a.getRows());
    return res < 0.3;
}

Matrix Matrix::operator==(Matrix &b)
{
    return equals(*this, b);
}

int Matrix::upperBandwidth()
{
    Matrix a = *this;
    int max = 0;
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = i + 1; j < a.cols; j++)
        {
            if (a.data[i][j].real() != 0)
            {
                max = std::max(max, j - i);
            }
        }
    }
    return max;
}

int Matrix::lowerBandwidth()
{
    Matrix a = *this;
    int max = 0;
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < i; j++)
        {
            if (a.data[i][j].real() != 0)
            {
                max = std::max(max, i - j);
            }
        }
    }
    return max;
}

float Matrix::bandDensity()
{
    Matrix a = *this;
    int count = 0;
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            if (a.data[i][j].real() != 0)
            {
                count++;
            }
        }
    }
    return static_cast<float>(count) / (a.getRows() * a.getCols());
}

void Matrix::bandedDecomposition(Matrix &A, Matrix &L, Matrix &U)
{
    int upperBandwidth = A.upperBandwidth();
    int lowerBandwidth = A.lowerBandwidth();

    if (upperBandwidth == 0 && lowerBandwidth == 0)
    {
        throw std::invalid_argument("Matrix is not banded");
    }

    L = Matrix(A.getRows());
    U = Matrix(A.getRows());

    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = std::max(0, i - lowerBandwidth); j <= std::min(A.getCols() - 1, i + upperBandwidth); j++)
        {
            if (j < i)
            {
                std::complex<double> sum = 0;
                for (int k = std::max(0, j - lowerBandwidth); k < j; k++)
                {
                    sum += L.get(i, k) * U.get(k, j);
                }
                L.set(i, j, (A.get(i, j) - sum) / U.get(j, j));
            }
            else
            {
                std::complex<double> sum = 0;
                for (int k = std::max(0, i - upperBandwidth); k < i; k++)
                {
                    sum += L.get(i, k) * U.get(k, j);
                }
                U.set(i, j, A.get(i, j) - sum);
            }
        }
    }
}

bool Matrix::positiveDiagonal()
{
    Matrix a = *this;
    for (int i = 0; i < a.getRows(); i++)
    {
        if (a.get(i, i).real() <= 0)
        {
            return false;
        }
    }
    return true;
}

bool Matrix::negativeDiagonal()
{
    Matrix a = *this;
    for (int i = 0; i < a.getRows(); i++)
    {
        if (a.get(i, i).real() >= 0)
        {
            return false;
        }
    }
    return true;
}

Matrix Matrix::bandedSolver(Matrix &b)
{
    Matrix a = *this;
    Matrix L(a.getRows());
    Matrix U(a.getRows());
    Matrix::bandedDecomposition(a, L, U);
    Matrix y = L.forwardSolve(b);
    Matrix x = U.backwardSolve(y);
    return x;
}

void Matrix::LDLTDecomposition(Matrix &A, Matrix &L, Matrix &D)
{
    if (!A.isHermitian())
    {
        throw std::invalid_argument("Matrix is not Hermitian");
    }

    L = Matrix(A.getRows());
    D = Matrix(A.getRows());

    for (int i = 0; i < A.getRows(); i++)
    {
        for (int j = 0; j <= i; j++)
        {
            std::complex<double> sum = 0;
            for (int k = 0; k < j; k++)
            {
                sum += L.get(i, k) * D.get(k, k) * std::conj(L.get(j, k));
            }
            if (i == j)
            {
                D.set(i, i, A.get(i, i) - sum);
            }
            else
            {
                L.set(i, j, (A.get(i, j) - sum) / D.get(j, j));
            }
        }
    }
}

Matrix Matrix::diagonalSolver(Matrix &b)
{
    Matrix a = *this;
    Matrix x(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++)
    {
        if (a.get(i, i).real() == 0)
        {
            throw std::invalid_argument("Matrix is singular");
        }
        x.set(i, 0, b.get(i, 0) / a.get(i, i));
    }
    return x;
}

Matrix Matrix::LDLTSolver(Matrix &b)
{
    Matrix a = *this;
    Matrix L(a.getRows());
    Matrix D(a.getRows());
    Matrix::LDLTDecomposition(a, L, D);
    Matrix y = L.forwardSolve(b);
    Matrix z = D.diagonalSolver(y);
    Matrix x = L.transpose().backwardSolve(z);
    return x;
}

bool Matrix::isDiagonal() {
    Matrix a = *this;
    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getCols(); j++) {
            if (i != j && a.get(i, j).real() != 0) {
                return false;
            }
        }
    }
    return true;
}

Matrix Matrix::operator/=(Matrix &b)
{
    Matrix a = *this;
    if(a.isSquare()) {
        if(!a.isSparse()) {
            return (a\b);
        }

    } else {
        return a.QRSolver(b);
    }
}