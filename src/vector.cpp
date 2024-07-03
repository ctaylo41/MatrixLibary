#include "vector.h"
#include "matrix.h"
#include <cmath>

std::vector<std::vector<double>> Vector::convertToMatrix(std::vector<double> data) {
    std::vector<std::vector<double>> temp;
    for (double& value : data) {
        temp.push_back(std::vector<double>{value});
    }
    return temp;
}

double Vector::operator*(Vector& b) {
    double result = 0;
    for (int i = 0; i < this->rows; i++) {
        result += this->data[i][0] * b.data[i][0];
    }
    return result;
}

double Vector::magnitude() {
    double result = 0;
    for (int i = 0; i < this->rows; i++) {
        result += this->data[i][0] * this->data[i][0];
    }
    return sqrt(result);
}

Matrix Vector::operator^(Vector& b) {
    if (this->rows != 3 || b.rows != 3) {
        throw "Cross product is only defined for 3D vectors";
    }
    std::vector<std::vector<double>> temp = {
        {this->data[1][0] * b.data[2][0] - this->data[2][0] * b.data[1][0]},
        {this->data[2][0] * b.data[0][0] - this->data[0][0] * b.data[2][0]},
        {this->data[0][0] * b.data[1][0] - this->data[1][0] * b.data[0][0]}
    };
    return Matrix(temp);
}