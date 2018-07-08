#ifndef ARRAY_H_INCLUDED
#define ARRAY_H_INCLUDED

template <class T>
class Array
{
    int arraySize = 0;
    T* object = nullptr;

    public:
        Array(){};
        Array(int size){arraySize = size; object = new T[arraySize];};
        ~Array(){if (object != nullptr) delete [] object;};

        int size(){return arraySize;}
        void setSize(int size)
        {
            // Create a new object array
            T* newObject = new T[size];

            // Copy over the old contents to the new one
            for (int i = 0; i < size && i < arraySize; i++)
                newObject[i] = object[i];

            // Delete the old array
            delete [] object;

            // Set the new object array reference
            object = newObject;

            // Update the size
            arraySize = size;
        }

        /* Operator overloading */
        void operator = (Array& otherArray)
        {
            /// Deep copy
            // Update the size
            arraySize = otherArray.size();

            // Delete the array if it already exists
            if (object != nullptr)
                delete [] object;

            // Create a new object array
            object = new T[arraySize];

            // Copy over the all contents
            for (int i = 0; i < arraySize; i++)
                object[i] = otherArray[i];
        }

        T& operator [] (int index)
        {
            // Selecting the correct index is the programmer's job.
            return object[index];
        }
};

#endif // ARRAY_H_INCLUDED
