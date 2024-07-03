#include "vector.h"
#include "matrix.h"
#include <cmath>

std::vector<std::vector<std::complex<double> > > Vector::convertToMatrix(std::vector<std::complex<double> > data) {
    std::vector<std::vector<std::complex<double> > > temp;
    for (std::complex<double>& value : data) {
        temp.push_back(std::vector<std::complex<double> >{value});
    }
    return temp;
}

std::complex<double> Vector::operator*(Vector& b) {
    std::complex<double> result = 0;
    for (int i = 0; i < this->rows; i++) {
        result += this->data[i][0] * b.data[i][0];
    }
    return result;
}

std::complex<double> Vector::magnitude() {
    std::complex<double> result = 0;
    for (int i = 0; i < this->rows; i++) {
        result += this->data[i][0] * this->data[i][0];
    }
    return sqrt(result);
}

Matrix Vector::operator^(Vector& b) {
    if (this->rows != 3 || b.rows != 3) {
        throw "Cross product is only defined for 3D vectors";
    }
    std::vector<std::vector<std::complex<double> > > temp = {
        {this->data[1][0] * b.data[2][0] - this->data[2][0] * b.data[1][0]},
        {this->data[2][0] * b.data[0][0] - this->data[0][0] * b.data[2][0]},
        {this->data[0][0] * b.data[1][0] - this->data[1][0] * b.data[0][0]}
    };
    return Matrix(temp);
}