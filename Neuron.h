#ifndef NEURON_H
#define NEURON_H

#include "Matrix.h"
#include "Array.h"
#include "EActivationFunction.h"

class Neuron
{
    public:
        Neuron();
        ~Neuron();

        void initWeightMatrix(int featureSize);
        void fillWeightMatrixRandomly(int featureSize, int minValue, int maxValue);
        void deltaLearning(Matrix<float> &featureMatrix, Array<float> &classificationVector, int epoch, float learningRate);
        void hebbianLearning(Matrix<float> &featureMatrix, int epoch, float learningRate);
        float predict(Matrix<float>& dataPoint); // Predicts the classification for the given data point

        void printWeightMatrix();

        float activationFunction(float input); // Relays the input to the function specified
        float derivedActivationFunction(float input);

        void getAugmentedDataSample(Matrix<float> &input, Matrix<float> &output);

        EActivationFunction activationFunctionEnum; // Specifies the learning response function to be used
        Matrix<float> weightMatrix; // Weight matrix of the perceptron
        float lastNetInput; // Weight matrix of the perceptron


    private:
        bool weightMatrixSet;
};

#endif // NEURON_H
