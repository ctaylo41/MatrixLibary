#include "matrix.h"
#include <vector>
class Vector : public Matrix {
    private:
        std::vector<std::vector<std::complex<double> > > convertToMatrix(std::vector<std::complex<double> > data);
        std::complex<double> one = 1;
    public:
        Vector(int n) : Matrix(n,1.0) {} ;
        Vector(std::vector<std::complex<double> > data) : Matrix(convertToMatrix(data)) {};
        Vector(Matrix &a);
        std::complex<double> operator*(Vector& b);
        Vector operator*(std::complex<double> b);
        Vector operator+(Vector& b);
        Vector operator-(Vector& b);
        std::complex<double> magnitude();
        Matrix operator^(Vector& b);
        Vector norm();
};