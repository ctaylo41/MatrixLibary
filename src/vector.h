#include "matrix.h"
#include <vector>
class Vector : public Matrix {
    private:
        std::vector<std::vector<double>> convertToMatrix(std::vector<double> data);
    public:
        Vector(int n) : Matrix(n,1.0) {} ;
        Vector(std::vector<double> data) : Matrix(convertToMatrix(data)) {};
        double operator*(Vector& b);
        double magnitude();
        Matrix operator^(Vector& b);
};