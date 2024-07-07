#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string>
#include <utility>
#include <complex>
class Matrix
{
protected:
    int rows;
    int cols;
    std::vector<std::vector<std::complex<double> > > data;

public:
    Matrix(int rows, int cols);
    Matrix(std::vector<std::vector<std::complex<double> > > data);
    Matrix transpose();
    Matrix inverse();
    Matrix choleskyDecomp();
    bool isSquare();
    Matrix operator-(Matrix &b);
    Matrix operator+(Matrix &b);
    Matrix operator*(Matrix &b);
    Matrix operator*(double b);
    Matrix operator/(Matrix &b);
    std::complex<double> get(int i, int j);
    int getRows();
    int getCols();
    Matrix getRow(int i);
    Matrix getCol(int i);
    void set(int i, int j, std::complex<double> value);
    std::vector<std::vector<std::complex<double> > > getData();
    bool equals(Matrix &a, Matrix &b);
    Matrix(int n);
    Matrix forwardSolve(Matrix &b);
    Matrix backwardSolve(Matrix &b);
    std::complex<double> determinant();
    static void LUDecomposition(Matrix &a, Matrix &L, Matrix &U, Matrix &P);
    std::string toString();
    bool isUpperHessenberg();
    bool isTriDiagonal();
    bool isHermitian();
    Matrix conjugate();
    Matrix QRSolver(Matrix &b);
    Matrix GramSchmidt();
    static void QRDecomposition(Matrix &A, Matrix &Q, Matrix &R);
    bool isUpperTriangular();
    bool isLowerTriangular();
    Matrix tridigonalSolver(Matrix &b);
};
#endif