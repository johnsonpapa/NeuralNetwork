#include "Neuron.h"
#include <math.h>
#include <iostream>
// #include <limits>

#define EULER_NUMBER 2.71828182845904523536

Neuron::Neuron()
{
    weightMatrixSet = false;
    activationFunctionEnum = HEAVISIDE;
}

Neuron::~Neuron()
{

}

void Neuron::initWeightMatrix(int featureSize)
{
    weightMatrix.setSize(featureSize + 1, 1);
//    weightMatrix.fill(0);
//    weightMatrix[0][0] = 1;
//    weightMatrix.fill(1);
//    weightMatrix[0][0] = 0;
    fillWeightMatrixRandomly(featureSize, -100, 100); // Apparently this helps
    weightMatrixSet = true;
}

void Neuron::fillWeightMatrixRandomly(int featureSize, int minValue, int maxValue)
{
    int range = (maxValue - minValue) * 1000;
    weightMatrix[0][0] = 1;
    for (int i = 1; i < featureSize + 1; i++)
        weightMatrix[i][0] = (float) (rand() % range) / 1000.0f + minValue;
    weightMatrixSet = true;
}

void Neuron::deltaLearning(Matrix<float> &featureMatrix, Array<float> &classificationVector, int epoch, float learningRate)
{
    /// 1) Check for any missed/erroneous parameters
    // Initialise the weight vector
    if (!weightMatrixSet)
        initWeightMatrix(featureMatrix.getSizeX());

    if (featureMatrix.getSizeX() != weightMatrix.getSizeX() - 1)
    {
        std::cout << "Delta learning: The feature dimensionality size in the feature matrix must equal to the weight matrix size!" << std::endl;
        return;
    }

    if (featureMatrix.getSizeX() <= 0)
    {
        std::cout << "Delta learning: The feature dimension must be larger than 0 for learning to occur!" << std::endl;
        return;
    }

    if (featureMatrix.getSizeY() != classificationVector.size())
    {
        std::cout << "Delta learning: The number of samples in the feature matrix must equal to the classification vector size!" << std::endl;
        return;
    }

    /// Proceed with the delta learning algorithm
    int featureDimension = featureMatrix.getSizeX();

    // Create the augmented data sample matrix (vector)
    Matrix<float> augmentedDataSample(1, featureDimension + 1); // Taken outside the loop to speed things up
    augmentedDataSample[0][0] = 1; // This value is always 1

    // For randomising the access function for Stochastic learning
//    std::vector<int> accessOrder;

    // Loop the delta learning rule epoch times
    for (int i = 0; i < epoch; i++)
    {
        // Add all entries to it
//        for (int i = 0; i < classificationVector.size(); i++)
//            accessOrder.push_back(i);

        // Loop through every single data sample
        for (int j = 0; j < classificationVector.size(); j++)
        {
            // Set the data for the augmented sample matrix (vector)
            for (int k = 0; k < featureDimension; k++)
                augmentedDataSample[0][k + 1] = featureMatrix[k][j];

            // Calculate the neuron response
            Matrix<float> resultMatrix;
            resultMatrix.dot(weightMatrix, augmentedDataSample);
            float response = activationFunction(resultMatrix[0][0]); // The result matrix should be of size(1, 1)

//            std::cout << "DELTA RULE LEARNING: Predicted " << resultMatrix[0][0] << " -> " << response << ", aim = " << classificationVector[0] << std::endl;

            // Update the weight with Delta update rule: w = w + n(t - y)x
            float factor = learningRate * (classificationVector[j] - response); // n(t - y)
            for (int k = 0; k < weightMatrix.getSizeX(); k++)
                weightMatrix[k][0] = weightMatrix[k][0] + factor * augmentedDataSample[0][k];
        }
    }
}

void Neuron::hebbianLearning(Matrix<float> &featureMatrix, int epoch, float learningRate)
{
    /// 1) Check for any missed/erroneous parameters
    // Initialise the weight vector
    if (!weightMatrixSet)
        initWeightMatrix(featureMatrix.getSizeX());

    if (featureMatrix.getSizeX() != weightMatrix.getSizeX() - 1)
    {
        std::cout << "Hebbian learning: The feature dimensionality size in the feature matrix must equal to the weight matrix size!" << std::endl;
        return;
    }

    if (featureMatrix.getSizeX() <= 0)
    {
        std::cout << "Hebbian learning: The feature dimension must be larger than 0 for learning to occur!" << std::endl;
        return;
    }

    /// Proceed with the delta learning algorithm
    int featureDimension = featureMatrix.getSizeX();

    // Create the augmented data sample matrix (vector)
    Matrix<float> augmentedDataSample(1, featureDimension + 1); // Taken outside the loop to speed things up
    augmentedDataSample[0][0] = 1; // This value is always 1

    // Loop the delta learning rule epoch times
    for (int i = 0; i < epoch; i++)
    {
        // Loop through every single data sample
        for (int j = 0; j < featureMatrix.getSizeY(); j++)
        {
            // Set the data for the augmented sample matrix (vector)
            for (int k = 0; k < featureDimension; k++)
                augmentedDataSample[0][k + 1] = featureMatrix[k][j];

            // Calculate the neuron response
            Matrix<float> resultMatrix;
            resultMatrix.dot(weightMatrix, augmentedDataSample);
            float response = activationFunction(resultMatrix[0][0]); // The result matrix should be of size(1, 1)

            // Update the weight with Delta update rule: w = w + nyx
            float factor = learningRate * response; // ny
            for (int k = 0; k < weightMatrix.getSizeX(); k++)
                weightMatrix[k][0] = weightMatrix[k][0] + factor * augmentedDataSample[0][k];
        }
    }
}

float Neuron::predict(Matrix<float>& dataPoint)
{
    if (!weightMatrixSet)
        initWeightMatrix(dataPoint.getSizeX());

    if (weightMatrix.getSizeX() != dataPoint.getSizeX() + 1) // +1 because the weight matrix is augmented
    {
        std::cout << "Incorrect number of feature dimension entered for data point. Got " << dataPoint.getSizeX() << ". Expected " << weightMatrix.getSizeX() - 1 << std::endl;
        return -1;
    }

    // Calculate the neuron response
    Matrix<float> dataMatrix; // Augment the data point
    getAugmentedDataSample(dataPoint, dataMatrix);
    Matrix<float> resultMatrix; // Create the result matrix to store the multiplied result
    resultMatrix.dot(weightMatrix, dataMatrix);
    lastNetInput = resultMatrix[0][0];
    return activationFunction(resultMatrix[0][0]);
}

float Neuron::activationFunction(float input)
{
    switch (activationFunctionEnum)
    {
        case LINEAR:
			return input;

		case HEAVISIDE: // 0 to 1
			if (input > 0) return 1;
			else if (input == 0) return 0.5f;
			else return 0;

		case LOGISTIC: // From 0 to 1
			return 1.0f / (1.0f + (float) pow(EULER_NUMBER, -input));

        case SOFTMAX:
            return 0;

		case TANH: // From -1 to 1
			return tanh(input);

		case TANH01: // From 0 to 1
			return tanh(input) / 2.0f + 0.5f;

        case RECTIFIED_LINEAR_UNIT:
            if (input < 0) return 0;
            return input;

		case ARCTAN: // From -pi/2 to pi/2
			return atan(input);

        case ARCTAN01: // From 0 to 1
			return atan(input) / 3.14159265358979f + 0.5f;

		case SYMMETRICAL_HARD_LIMIT: // -1 to 1
			if (input > 0) return 1;
			else if (input == 0) return 0;
			else return -1;

		case SINUSOID: // -1 to 1
			return sin(input);

        case SINUSOID01: // 0 to 1
			return sin(input) / 2.0f + 0.5f;

		case GAUSSIAN: // 0 to 1
			return pow(EULER_NUMBER, -input * input);

		default:
			return input; // Assume linear otherwise
    }
}

float Neuron::derivedActivationFunction(float input)
{
    switch (activationFunctionEnum)
    {
		case LINEAR:
			return 1;

		case HEAVISIDE:
		    // if (input == 0) return std::numeric_limits<float>::infinity();
			return 0; // Will not work with back propagation

		case LOGISTIC:
			return atan(input) * (1.0f - atan(input)) / 2.0f;

        case SOFTMAX:
            return 0;

		case TANH:
			return 1.0f - pow(activationFunction(input), 2.0f);

		case TANH01:
			return (1.0f - pow(activationFunction(input), 2.0f)) / 2.0f;

        case RECTIFIED_LINEAR_UNIT:
            if (input < 0) return 0;
            return 1;

		case ARCTAN:
			return 1.0f / (input * input + 1.0f);

        case ARCTAN01:
			return 1.0f / (input * input + 1.0f) / 3.14159265358979f; // Unsure here

		case SYMMETRICAL_HARD_LIMIT:
			return 0;

		case SINUSOID:
			return cos(input);

		case SINUSOID01:
			return cos(input) / 2.0f;

		case GAUSSIAN:
			return -2 * input * pow(EULER_NUMBER, -input * input);

		default:
			return 1; // Assume linear otherwise
    }
}

void Neuron::printWeightMatrix()
{
    for (int i = 0; i < weightMatrix.getSizeX(); i++)
        std::cout << weightMatrix[i][0] << " ";
    std::cout << std::endl;
}

void Neuron::getAugmentedDataSample(Matrix<float>& input, Matrix<float>& output)
{
    output.setSize(1, input.getSizeX() + 1); // Taken outside the loop to speed things up
    output[0][0] = 1; // This value is always 1
    for (int i = 0; i < input.getSizeX(); i++)
        output[0][i + 1] = input[i][0];
}
