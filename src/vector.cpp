#include "vector.h"
#include "matrix.h"
#include <cmath>

// Constructor for Vector class
// @param n: int - number of rows

std::vector<std::vector<std::complex<double> > > Vector::convertToMatrix(std::vector<std::complex<double> > data) {
    std::vector<std::vector<std::complex<double> > > temp;
    for (std::complex<double>& value : data) {
        temp.push_back(std::vector<std::complex<double> >{value});
    }
    return temp;
}
// Dot product of two vectors
// @param b: Vector - vector to dot with
// @return std::complex<double> - dot product of two vectors
std::complex<double> Vector::operator*(Vector& b) {
    std::complex<double> result = 0;
    for (int i = 0; i < this->rows; i++) {
        result += this->data[i][0] * b.data[i][0];
    }
    return result;
}
// Scalar multiplication of a vector
// @param b: std::complex<double> - scalar to multiply by
// @return Vector - vector multiplied by scalar
Vector Vector::operator*(std::complex<double> b) {
    std::vector<std::complex<double> >  temp;
    for (int i = 0; i < this->rows; i++) {
        temp.push_back(this->data[i][0] * b);
    }
    return Vector(temp);
}

// Addition of two vectors
// @param b: Vector - vector to add
// @return Vector - sum of two vectors
Vector Vector::operator+(Vector& b) {
    std::vector<std::complex<double> > temp;
    for (int i = 0; i < this->rows; i++) {
        temp.push_back(this->data[i][0] + b.data[i][0]);
    }
    return Vector(temp);
}
// Subtraction of two vectors
// @param b: Vector - vector to subtract
// @return Vector - difference of two vectors
Vector Vector::operator-(Vector& b) {
    std::vector<std::complex<double> > temp;
    for (int i = 0; i < this->rows; i++) {
        temp.push_back(this->data[i][0] - b.data[i][0]);
    }
    return Vector(temp);
}

// Magnitude of a vector
// @return std::complex<double> - magnitude of vector
std::complex<double> Vector::magnitude() {
    std::complex<double> result = 0;
    for (int i = 0; i < this->rows; i++) {
        result += this->data[i][0] * this->data[i][0];
    }
    return sqrt(result);
}
// Cross product of two vectors
// @param b: Vector - vector to cross with
// @return Matrix - cross product of two vectors
// @throws char* - if vectors are not 3D
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
// Normalize a vector
// @return Vector - normalized vector
Vector Vector::norm() {
    return *this * (this->one.real() / this->magnitude().real());
}