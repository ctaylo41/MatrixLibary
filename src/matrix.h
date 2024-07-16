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
    bool complex;

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
    Matrix operator==(Matrix &b);
    std::complex<double> get(int i, int j);
    int getRows();
    int getCols();
    Matrix getRow(int i);
    Matrix getCol(int i);
    void set(int i, int j, std::complex<double> value);
    void setRow(int i, Matrix &row);
    void setCol(int i, Matrix &col);
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
    Matrix calculateGivenRotationMatrix(int i,int j);
    static void hessenbergQRDecomposition(Matrix &A,Matrix &Q, Matrix &R);
    Matrix hessenbergSolver(Matrix &b);
    Matrix LUSolver(Matrix &b);
    bool isSparse();
    int upperBandwidth();
    int lowerBandwidth();
    float bandDensity();
    static void bandedDecomposition(Matrix &A, Matrix &L, Matrix &U);
    Matrix bandedSolver(Matrix &b);
    static void LDLTDecomposition(Matrix &A, Matrix &L, Matrix &D);
    bool positiveDiagonal();
    bool negativeDiagonal();
};
#endif