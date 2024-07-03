#include <gtest/gtest.h>
#include "../src/matrix.h" 
#include <vector>
TEST(MatrixTest, HandlesConstruction) {
    Matrix m(2, 3);
    std::vector<std::vector<std::complex<double> > > data = {{1,2,3},{4,5,6}};
    Matrix m2(data);
    EXPECT_EQ(m.getRows(), 2);
    EXPECT_EQ(m.getCols(), 3);
    EXPECT_EQ(m2.getRows(), 2);
    EXPECT_EQ(m2.getCols(), 3);
    for(int i = 0; i < m.getRows(); i++){
        for(int j = 0; j < m.getCols(); j++){
            EXPECT_EQ(m2.get(i, j), data[i][j]);
        }
    }   
}

TEST(MatrixTes, HandlesTranspose) {
    std::vector<std::vector<std::complex<double> > > data = {{1,2,3},{4,5,6}};
    Matrix m(data);
    Matrix m2 = m.transpose();
    for(int i = 0; i < m.getRows(); i++){
        for(int j = 0; j < m.getCols(); j++){
            EXPECT_EQ(m.get(i, j), m2.get(j, i));
        }
    }
    
    data = {{1,2,3}};
    m = Matrix(data);
    m2 = m.transpose();
    for(int i = 0; i < m.getRows(); i++){
        for(int j = 0; j < m.getCols(); j++){
            EXPECT_EQ(m.get(i, j), m2.get(j, i));
        }
    }
}





// Test the get method
TEST(MatrixTest, HandlesGet) {
    Matrix m(2, 2);
    // Assuming you have a method to set values for testing
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(1, 0, 3.0);
    m.set(1, 1, 4.0);
    EXPECT_EQ(m.get(0, 0), 1.0);
    EXPECT_EQ(m.get(1, 1), 4.0);
}

// Test the Multiply method
TEST(MatrixTest, HandlesMultiply) {
    std::vector<std::vector<std::complex<double> > > data1 = {{1, 2}, {3, 4}};
    std::vector<std::vector<std::complex<double> > > data2 = {{5, 6}, {7, 8}};
    Matrix a(data1);
    Matrix b(data2);

    // Populate a and b with values...
    Matrix result = a*b;
    std::vector<std::vector<std::complex<double> > > res = {{19, 22}, {43, 50}};
    Matrix expected(res);
    bool equal = a.equals(result, expected);
    EXPECT_TRUE(equal);
}

TEST(MatrixTest, HandlesCholesky) {
    std::vector<std::vector<std::complex<double> > > data = {{25, 15, -5}, {15, 18, 0}, {-5, 0, 11}};
    Matrix a(data);
    Matrix result = a.choleskyDecomp();
    std::vector<std::vector<std::complex<double> > > res = {{5, 0, 0}, {3, 3, 0}, {-1, 1, 3}};
    Matrix expected(res);
    bool equal = a.equals(result, expected);
    EXPECT_TRUE(equal);
    Matrix a2 = expected.transpose();
    Matrix a3 = expected*a2;
    EXPECT_TRUE(a.equals(a3, a));
}


TEST(MatrixTest, HandlesInverse) {
    std::vector<std::vector<std::complex<double> > > data = {{4, 1, 1}, {1, 3, -1}, {1, -1, 3}};
    Matrix a(data);
    Matrix result = a.inverse();
    std::vector<std::vector<std::complex<double> > > res = {{1.0/3, -1.0/6, -1.0/6}, {-1.0/6, 11.0/24, 5.0/24}, {-1.0/6, 5.0/24, 11.0/24}};
    Matrix expected(res);
    bool equal = a.equals(result, expected);
    EXPECT_TRUE(equal);
    Matrix idenity(3);
    Matrix a2 = a*result;
    EXPECT_TRUE(a.equals(a2, idenity));
}




TEST(MatrixTest, HandlesForwardSolve) {
    std::vector<std::vector<std::complex<double> > > data = {{1, 0, 0}, {2, 1, 0}, {3, 4, 1}};
    Matrix a(data);
    std::vector<std::vector<std::complex<double> > > data2 = {{1}, {2}, {3}};
    Matrix b(data2);
    Matrix result = a.forwardSolve(b);
    std::vector<std::vector<std::complex<double> > > res = {{1}, {0}, {0}};
    Matrix expected(res);
    bool equal = a.equals(result, expected);
    EXPECT_TRUE(equal);
}

TEST(MatrixTest, HandlesBackwardSolve) {
    std::vector<std::vector<std::complex<double> > > data = {{1, 2, 3}, {0, 1, 4}, {0, 0, 1}};
    Matrix a(data);
    std::vector<std::vector<std::complex<double> > > data2 = {{14}, {8}, {3}};
    Matrix b(data2);
    Matrix result = a.backwardSolve(b);
    std::vector<std::vector<std::complex<double> > > res = {{13}, {-4}, {3}};
    Matrix expected(res);
    bool equal = a.equals(result, expected);
    EXPECT_TRUE(equal);
}

TEST(MatrixTest, LUDecomp) {
    std::vector<std::vector<std::complex<double> > > data = {{1, 2, 3}, {2, 5, 3}, {1, 0, 8}};
    Matrix a(data);
    Matrix L(3);
    Matrix U(3);
    Matrix P(3);
    Matrix::LUDecomposition(a, L, U, P);
    std::vector<std::vector<std::complex<double> > > lower = {{1, 0, 0}, {0.5, 1, 0}, {0.5, 0.2,1}};
    std::vector<std::vector<std::complex<double> > > upper = {{2, 5, 3}, {0, -2.5, 6.5}, {0, 0, 0.2}};
    std::vector<std::vector<std::complex<double> > > perm = {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}};
    Matrix L2(lower);
    Matrix U2(upper);
    Matrix P2(perm);
    EXPECT_TRUE(a.equals(L, L2));
    EXPECT_TRUE(a.equals(U, U2));
    EXPECT_TRUE(a.equals(P, P2));
    
}

TEST(MatrixTest, HandlesDeterminant) {
    std::vector<std::vector<std::complex<double> > > data = {{1, 2, 3}, {2, 5, 3}, {1, 0, 8}};
    Matrix a(data);
    std::complex<double> result = a.determinant();
    std::complex<double> expected = -1.0;
    EXPECT_DOUBLE_EQ(result.real(), expected.real());
    EXPECT_DOUBLE_EQ(result.imag(), expected.imag());
}


TEST(MatrixTest, HandlesAddition) {
    std::vector<std::vector<std::complex<double> > > data = {{1, 2, 3}, {2, 5, 3}, {1, 0, 8}};
    Matrix a(data);
    Matrix b(data);
    Matrix result = a+b;
    std::vector<std::vector<std::complex<double> > > res = {{2, 4, 6}, {4, 10, 6}, {2, 0, 16}};
    Matrix expected(res);
    bool equal = a.equals(result, expected);
    EXPECT_TRUE(equal);
}

TEST(MatrixTest, HandlesSubtraction) {
    std::vector<std::vector<std::complex<double> > > data = {{1, 2, 3}, {2, 5, 3}, {1, 0, 8}};
    Matrix a(data);
    Matrix b(data);
    Matrix result = a-b;
    std::vector<std::vector<std::complex<double> > > res = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Matrix expected(res);
    bool equal = a.equals(result, expected);
    EXPECT_TRUE(equal);
}

TEST(MatrixTest, HandlesScalarMultiplication) {
    std::vector<std::vector<std::complex<double> > > data = {{1, 2, 3}, {2, 5, 3}, {1, 0, 8}};
    Matrix a(data);
    Matrix result = a*2;
    std::vector<std::vector<std::complex<double> > > res = {{2, 4, 6}, {4, 10, 6}, {2, 0, 16}};
    Matrix expected(res);
    bool equal = a.equals(result, expected);
    EXPECT_TRUE(equal);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}