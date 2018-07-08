#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <memory>
#include <math.h>
#include <iostream>

/**
    UPDATE: 8/7/2018
    How element access works:
    Horizontal = x = columns
    Vertical = y = rows

    Given (i, j), access becomes: i * sizeY + j.
    A little counter intuitive but actually makes more sense this way
*/
template <class T>
class Matrix
{
    std::shared_ptr<T[]> ptr;
    int sizeX, sizeY;

    public:
        Matrix(){sizeX = 0; sizeY = 0;}
        Matrix(int _sizeX, int _sizeY) : sizeX(_sizeX), sizeY(_sizeY)
            {ptr.reset(new T[sizeX * sizeY]);};
        ~Matrix(){};

        // Methods
        int getSizeX() const {return sizeX;}
        int getSizeY() const {return sizeY;}
        int getSize() const { return sizeX * sizeY;}

        /**
        Matrix element access method
        */
        T& getElement(int x, int y)
        {
            return ptr[x * sizeY + y];
        }

        /**
        Creates and returns a sub matrix given the dimensions
        relative to the current matrix
        */
        Matrix<T> subMatrix(int minX, int maxX, int minY, int maxY)
        {
            /// 1) Limit the parameters
            if (minX < 0) minX = 0;
            if (maxX >= sizeX) maxX = sizeX - 1;
            if (minY < 0) minY = 0;
            if (maxY >= sizeY) maxY = sizeY - 1;

            /// 2) Create the sub matrix
            Matrix<T> matrix(maxX - minX + 1, maxY - minY + 1);
            for (int x = 0; x < matrix.getSizeX(); x++)
                for (int y = 0; y < matrix.getSizeY(); y++)
                    matrix[x][y] = (*this)[x + minX][y + minY];

            /// 3) Return the sub matrix
            return matrix;
        }

        /**
        Fills the entire matrix with the specified value
        */
        void fill(T value)
        {
            for (int i = 0; i < sizeX * sizeY; i++)
                ptr[i] = value;
        }

        /// Operators
        T* operator [] (int index)
        {
            return &ptr[index * sizeY];
        }

        void setSize(int newSizeX, int newSizeY)
        {
            T* newArray = new T[newSizeX * newSizeY];

            // Copy over the array contents of the current array to the new array.
            for (int y = 0; y < newSizeY && y < sizeY; y++)
                for (int x = 0; x < newSizeX && x < sizeX; x++)
                {
                    newArray[(x * newSizeY) + y] = ptr[(x * sizeY) + y];
                }

            // Delete the old array as it is no longer needed
            ptr.reset(newArray);
            sizeX = newSizeX;
            sizeY = newSizeY;
        }

        T* getArrayRef()
        {
            return ptr.get();
        }

//        void operator= (Matrix<T> &matrix) // Deep copy for same typed matrices
        void operator= (Matrix<T> matrix) // Deep copy for same typed matrices
        {
            sizeX = matrix.getSizeX();
            sizeY = matrix.getSizeY();
            ptr.reset(new T[sizeX * sizeY]);
            for (int y = 0; y < sizeY; y++)
                for (int x = 0; x < sizeX; x++)
                    ptr[(x * sizeY) + y] = matrix[x][y];
        }

        void operator= (Matrix<T> *matrix) // Deep copy for same typed matrices
        {
            sizeX = matrix->getSizeX();
            sizeY = matrix->getSizeY();
            ptr.reset(new T[sizeX * sizeY]);
            for (int y = 0; y < sizeY; y++)
                for (int x = 0; x < sizeX; x++)
                    ptr[(x * sizeY) + y] = (*matrix)[x][y];
        }

        Matrix<T> operator + (Matrix<T> matrix)
        {
            if (sizeX == matrix.getSizeX() && sizeY == matrix.getSizeY())
            {
                Matrix<T> toReturn;
                toReturn.setSize(sizeX, sizeY);
                for (int x = 0; x < sizeX ; x++)
                    for (int y = 0; y < sizeY ; y++)
                        toReturn[x][y] = (*this)[x][y] + matrix[x][y];
                return toReturn;
            }
            else
            {
                std::cout << "Matrix passed for addition operator is of different size than expected! (" << sizeX <<  ", " << sizeY << ") != (" << matrix.getSizeX() <<  ", " << matrix.getSizeY() << ")" << std::endl;
                throw std::invalid_argument("Matrix passed for addition operator is of different size than expected!");
            }
        }

        Matrix<T> operator - (Matrix<T> matrix)
        {
            if (sizeX == matrix.getSizeX() && sizeY == matrix.getSizeY())
            {
                Matrix<T> toReturn;
                toReturn.setSize(sizeX, sizeY);
                for (int x = 0; x < sizeX ; x++)
                    for (int y = 0; y < sizeY ; y++)
                        toReturn[x][y] = (*this)[x][y] - matrix[x][y];
                return toReturn;
            }
            else
            {
                std::cout << "Matrix passed for addition operator is of different size than expected! (" << sizeX <<  ", " << sizeY << ") != (" << matrix.getSizeX() <<  ", " << matrix.getSizeY() << ")" << std::endl;
                throw std::invalid_argument("Matrix passed for addition operator is of different size than expected!");
            }
        }

        /// Numerical operations on matrices
        /**
        Produces the result of adding both matrices given that they apply
        */
        void add(Matrix<T> &matrix1, Matrix<T> &matrix2)
        {
            // Check if both matrices have the same dimensions
            if (matrix1.getSizeX() == matrix2.getSizeX() && matrix1.getSizeY() == matrix2.getSizeY())
            {
                // Reassign memory space for the pointer to match the size of the two matrices
                sizeX = matrix1.getSizeX();
                sizeY = matrix1.getSizeY();
                ptr.reset(new T[sizeX * sizeY]);

                // Perform addition on both matrices to this matrix
                for (int x = 0; x < sizeX ; x++)
                {
                    for (int y = 0; y < sizeY ; y++)
                    {
                        ptr[x * sizeY + y] = matrix1[x][y] + matrix2[x][y];
                    }
                }
            }
        }

        /**
        Produces the result of adding both matrices given that they apply
        */
        void deduct(Matrix<T> &matrix1, Matrix<T> &matrix2)
        {
            // Check if both matrices have the same dimensions
            if (matrix1.getSizeX() == matrix2.getSizeX() && matrix1.getSizeY() == matrix2.getSizeY())
            {
                // Reassign memory space for the pointer to match the size of the two matrices
                sizeX = matrix1.getSizeX();
                sizeY = matrix1.getSizeY();
                ptr.reset(new T[sizeX * sizeY]);

                // Perform addition on both matrices to this matrix
                for (int x = 0; x < sizeX ; x++)
                {
                    for (int y = 0; y < sizeY ; y++)
                    {
                        ptr[x * sizeY + y] = matrix1[x][y] - matrix2[x][y];
                    }
                }
            }
        }

        /**
        Produces the result of multiplying the current matrix with the given scalar value
        O(n) complexity
        */
        void multiply(T value)
        {
            for (int x = 0; x < sizeX; x++)
            {
                for (int y = 0; y < sizeY; y++)
                {
                    ptr[x * sizeY + y] *= value;
                }
            }
        }

        /**
        Dot product
        Produces the result of multiplying both matrices given that they apply
        Algorithm: Naive multiplication O(n^3)
        */
        void dot(Matrix<T> matrix1, Matrix<T> matrix2)
        {
            // Checking if the both matrices are compatible for multiplication
            if (matrix1.getSizeX() == matrix2.getSizeY())
            {
                // Reassign memory space for the pointer to match the size of the two matrices
                int x1 = matrix1.getSizeX();
                int x2 = matrix2.getSizeX();
                int y1 = matrix1.getSizeY();
                // int y2 = matrix2.getSizeY();
                ptr.reset(new T[x2 * y1]);
                sizeX = x2;
                sizeY = y1;

                // Perform multiplication on both matrices
                for (int i1 = 0; i1 < y1; i1++)
                {
                    for (int i2 = 0; i2 < x2; i2++)
                    {
                        T sum = 0;
                        for (int i3 = 0; i3 < x1; i3++)
                            sum += matrix1[i3][i1] * matrix2[i2][i3];

                        (*this)[i2][i1] = sum;
                    }
                }
            }
            else
            {
                std::cout << "Matrix dot operator cannot be performed due to incompatible matrices. " << matrix1.getSizeX() << " != " << matrix2.getSizeY() << std::endl;
            }
        }

        /**
        Transposes the matrix
        */
        void transpose()
        {
            // Create an temporary array
            int size = sizeX * sizeY;
            T tempArray[size];
            for (int i = 0; i < size; i++)
                tempArray[i] = ptr[i];

            // Then simply change the layout of everything
            for (int y = 0; y < sizeY; y++)
                for (int x = 0; x < sizeX; x++)
                    ptr[y * sizeY + x] = tempArray[x * sizeY + y];

            // Reverse the sizes
            sizeX = sizeY;
            sizeY = size / sizeX;
        }

        /**
        Clear the matrix by setting all values to 0
        */
        void clear()
        {
            for (int x = 0; x < sizeX; x++)
                {
                    for (int y = 0; y < sizeY; y++)
                    {
                        ptr[x * sizeY + y] = 0;
                    }
                }
        }

        /**
        Identity matrix function, transforms the current matrix into
        an identity matrix of specified size
        */
        void toIdentityMatrix(int size)
        {
            // If the size specified is valid (larger than 0
            if (size > 0)
            {
                ptr.reset(new T[size * size]);
                sizeX = size;
                sizeY = size;

                clear();
                for (int i = 0; i < size; i++)
                {
                    ptr[i * sizeY + i] = 1;
                }
            }
        }
};

#endif // MATRIX_H_INCLUDED
