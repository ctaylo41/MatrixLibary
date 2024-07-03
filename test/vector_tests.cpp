#include <gtest/gtest.h>
#include "../src/matrix.h" 
#include <vector>
#include "../src/vector.h"

TEST(VectorTest, HandlesConstruction) {
    Vector v(2);
    std::vector<double> data = {1,2};
    Vector v2(data);
    EXPECT_EQ(v.getRows(), 2);
    EXPECT_EQ(v.getCols(), 1);
    EXPECT_EQ(v2.getRows(), 2);
    EXPECT_EQ(v2.getCols(), 1);
    for(int i = 0; i < v.getRows(); i++){
        EXPECT_EQ(v2.get(i, 0), data[i]);
    }   
}

TEST(VectorTest, HandlesDotProduct) {
    std::vector<double> data1 = {1, 2, 3};
    std::vector<double> data2 = {4, 5, 6};
    Vector a(data1);
    Vector b(data2);
    EXPECT_EQ(a*b, 32);
}

TEST(VectorTest, HandlesMagnitude) {
    std::vector<double> data = {1, 2, 3};
    Vector a(data);
    EXPECT_EQ(a.magnitude(), sqrt(14));
}

TEST(VectorTest, HandlesCrossProduct) {
    std::vector<double> data1 = {1, 2, 3};
    std::vector<double> data2 = {4, 5, 6};
    Vector a(data1);
    Vector b(data2);
    Matrix result = a^b;
    std::vector<std::vector<double>> expected = {{-3},{6},{-3}};
    for(int i = 0; i < result.getRows(); i++){
        for(int j = 0; j < result.getCols(); j++){
            EXPECT_EQ(result.get(i, j), expected[i][j]);
        }
    }
}

