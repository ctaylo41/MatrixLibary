#include "matrix.h"
#include <vector>
class Vector : public Matrix {
    private:
        std::vector<std::vector<std::complex<double> > > convertToMatrix(std::vector<std::complex<double> > data);
    public:
        Vector(int n) : Matrix(n,1.0) {} ;
        Vector(std::vector<std::complex<double> > data) : Matrix(convertToMatrix(data)) {};
        std::complex<double> operator*(Vector& b);
        std::complex<double> magnitude();
        Matrix operator^(Vector& b);
};