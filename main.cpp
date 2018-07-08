#include <iostream>
#include <random>
#include <chrono>
#include <vector>

#include "Neuron.h"
#include "NeuralNetwork.h"

using namespace std;

void dataGenerator(int numberOfSamples, Matrix<float> &featureMatrix, Array<float> &classificationVector)
{
    // Set the matrix sizes
    int dimensionality = 2;
    featureMatrix.setSize(dimensionality, numberOfSamples);
    classificationVector.setSize(numberOfSamples); // 1D column vector

    // cout << "Sample data:" << endl;

    for (int i = 0; i < numberOfSamples; i++)
    {
        // Set the feature matrix sample data range
        for (int j = 0; j < dimensionality; j++)
            featureMatrix[j][i] = rand() % 1001 - 500;

        // The real classification function
        if (featureMatrix[0][i] - featureMatrix[1][i] >= 0) // Condition: x - y >= 0
            classificationVector[i] = 1;
        else // Class 0 otherwise
            classificationVector[i] = 0;
    }
}

void nonLinearDataGenerator(int numberOfSamples, Matrix<float> &featureMatrix, Array<float> &classificationVector)
{
    // Set the matrix sizes
    featureMatrix.setSize(3, numberOfSamples); // 1D samples
    classificationVector.setSize(numberOfSamples); // 1D vector (y direction)

    // cout << "Sample data:" << endl;

    for (int i = 0; i < numberOfSamples; i++)
    {
        // Set the feature matrix sample data range
        featureMatrix[0][i] = (float) (rand() % 10001) / 100.0f - 50.0f; // -50 <= x <= 50
        featureMatrix[1][i] = (float) (rand() % 10001) / 100.0f - 50.0f; // -50 <= y <= 50
        featureMatrix[2][i] = (float) (rand() % 10001) / 100.0f - 50.0f; // -50 <= z <= 50

        // The real classification function
        if (featureMatrix[0][i] * featureMatrix[0][i] + featureMatrix[0][i] * featureMatrix[1][i] - featureMatrix[1][i] * featureMatrix[1][i] + featureMatrix[1][i] * featureMatrix[2][i] >= 0) // Condition: xx + xy - yy + yz>= 0 -> Class 1
//        if (featureMatrix[0][i] * featureMatrix[0][i] - featureMatrix[1][i] * featureMatrix[1][i] - featureMatrix[2][i] * featureMatrix[2][i] >= 0) // xx - yy - zz >= 0
//        if (featureMatrix[0][i] - featureMatrix[1][i] + featureMatrix[2][i] >= 0) // x - y + z >= 0
            classificationVector[i] = 1;
        else // Class 0 otherwise
            classificationVector[i] = 0;
    }
}

void nonLinearSingleDataGenerator(Matrix<float> &featureMatrix, Array<float> &classificationVector)
{
    // Set the matrix sizes
    featureMatrix.setSize(1, 1); // 1D samples
    classificationVector.setSize(2); // 2 output classifications


    // Set the feature matrix sample data range
    featureMatrix[0][0] = (float) (rand() % 10001) / 100.0f - 50.0f; // -50 <= x <= 50

    // The real classification function
    if (sin(featureMatrix[0][0]) >= 0) // sin(x)
    {
        classificationVector[0] = 0;
        classificationVector[1] = 1;
    }
    else // Class 0 otherwise
    {
        classificationVector[0] = 1;
        classificationVector[1] = 0;
    }
}

float perceptronTest()
{
    cout << "### Neural network: Neuron test ###" << endl;
    Neuron perceptron;
    perceptron.activationFunctionEnum = TANH01;
    // perceptron.activationFunctionEnum = LOGISTIC;
    // perceptron.activationFunctionEnum = TANH; // Not sure why this doesn't work
    Matrix<float> featureMatrix;
    Array<float> classificationVector;
    dataGenerator(500, featureMatrix, classificationVector);

    /* Learning phase */
    cout << "\nLearning phase:" << endl;
    perceptron.deltaLearning(featureMatrix, classificationVector, 50, 0.5f);
    cout << "Success." << endl;
    // perceptron.hebbianLearning(featureMatrix, 50, 0.5);

    /* Data classification with training data */
    cout << "\nTraining data classification phase" << endl;
    int correct = 0;
    for (int i = 0; i < classificationVector.size(); i++)
    {
        Matrix<float> dataPoint = featureMatrix.subMatrix(0, featureMatrix.getSizeX(), i, i);
        if (round(perceptron.predict(dataPoint)) == classificationVector[i])
            correct++;
    }
    cout << "Correctly classified = " << correct << ", incorrectly classified = " << classificationVector.size() - correct << endl;

    /* Data prediction with new data samples */
    cout << "\nNew data testing phase" << endl;
    Matrix<float> testFeatureMatrix; // Generate test data
    Array<float> testClassificationMatrix;
    dataGenerator(100, testFeatureMatrix, testClassificationMatrix);
    correct = 0;
    for (int i = 0; i < testClassificationMatrix.size(); i++)
    {
        Matrix<float> dataPoint = testFeatureMatrix.subMatrix(0, testFeatureMatrix.getSizeX(), i, i);
        if (round(perceptron.predict(dataPoint)) == testClassificationMatrix[i])
            correct++;
    }
    cout << "Correctly classified = " << correct << ", incorrectly classified = " << testClassificationMatrix.size() - correct << endl;
    cout << "Neuron success rate = " << correct / (float) testClassificationMatrix.size() * 100 << "%" << endl;
    perceptron.printWeightMatrix();

    return correct / (float) testClassificationMatrix.size() * 100; // Return the success rate
}

float neuralNetwork(EActivationFunction hidden, EActivationFunction output, bool printStuff)
{
    int inputLayerSize = 2;
    int hiddenLayerSize = 1;
    int outputLayerSize = 1;
    int numberOfHiddenLayers = 1;
    NeuralNetwork neuralNetwork(inputLayerSize, hiddenLayerSize, outputLayerSize, numberOfHiddenLayers);
    neuralNetwork.learningRate = 0.05f;
    neuralNetwork.layers[0].setActivationFunction(LINEAR);
    neuralNetwork.layers[1].setActivationFunction(hidden);
    neuralNetwork.layers[2].setActivationFunction(output);

    Matrix<float> featureMatrix;
    Array<float> classificationVector;

    // Learning
    int learningSize = 5000;
    for (int i = 0; i < learningSize; i++)
    {
        dataGenerator(1, featureMatrix, classificationVector);
        // nonLinearDataGenerator(1, featureMatrix, classificationVector);
        // nonLinearSingleDataGenerator(featureMatrix, classificationVector);
        neuralNetwork.backpropagation(featureMatrix, classificationVector);
    }

    // Testing
    int testingSize = 1000;
    int correct = 0;
    for (int i = 0; i < testingSize; i++)
    {
        dataGenerator(1, featureMatrix, classificationVector);
        // nonLinearDataGenerator(1, featureMatrix, classificationVector);
        // nonLinearSingleDataGenerator(featureMatrix, classificationVector);
        neuralNetwork.forwardPropagation(featureMatrix);
        float response = neuralNetwork.getMaxResponse();
        if (round(response) == classificationVector[0])
            correct++;
    }

    if (printStuff)
    {
        cout << "### Neural network ###\nInput size = " << inputLayerSize
            << "\nHidden size = " << hiddenLayerSize
            << "\nOutput size = " << outputLayerSize
            << "\nNumber of hidden layers = " << numberOfHiddenLayers << std::endl;
        cout << "Correctly classified = " << correct << ", incorrectly classified = " << testingSize - correct << endl;
        cout << "Neural network success rate = " << (float) correct / (float) testingSize * 100.0f << "%" << endl;
    }
    return (float) correct / (float) testingSize * 100.0f;
}

void neuralNetworkFun()
{
    /* Iteratively find the best combination of inputs */
    std::vector<float> goodCombinations1, goodCombinations2;
    float limit = 70; // Min success rate to be added to good combinations
    int numberOfTests = 3;
    for (int x = 0; x < NOT_SPECIFIED; x++)
    {
        for (int y = 0; y < NOT_SPECIFIED; y++)
        {
            float successRate = 0;
            for (int i = 0; i < numberOfTests; i++)
            {
                successRate += neuralNetwork((EActivationFunction)x, (EActivationFunction)y, false);
            }
            successRate /= numberOfTests;

            // Requirement: Output the correct result
            if (successRate > limit)
            {
                goodCombinations1.push_back(x);
                goodCombinations2.push_back(y);
            }
            std::cout << x << " and " << y << ": The average success rate = " << successRate << std::endl;
        }
    }

    /// Good combinations
    std::cout << "##### Good combinations giving >=" << limit << "% accuracy: ";
    if (goodCombinations1.size() == 0)
        std::cout << "None.";
    else
    {
        for (int i = 0; i < goodCombinations1.size(); i++)
            std::cout << "<" << goodCombinations1[i] << "," << goodCombinations2[i] << ">   ";
    }
    std::cout << std::endl;

    // Get average, min and max for a all good combination of activation functions
    numberOfTests = 100;
    float successRate = 0;
    for (int i = 0; i <goodCombinations1.size(); i++)
    {
        successRate = 0;
        int x = goodCombinations1[i];
        int y = goodCombinations2[i];
        float min = 100;
        float max = 0;
        for (int i = 0; i < numberOfTests; i++)
        {
            float result = neuralNetwork((EActivationFunction) x, (EActivationFunction) y, false);
            if (result < min) min = result;
            if (result > max) max = result;
            successRate += result;
        }
        std::cout << x << " and " << y << ":" << std::endl;
        std::cout << "The average success rate = " << successRate / numberOfTests << std::endl;
        std::cout << "Min success rate = " << min << std::endl;
        std::cout << "Max success rate = " << max << std::endl << std::endl;
    }
}

int main()
{
    /* Initialisation */
    srand(time(NULL));

    // Neural network test 1
    neuralNetwork(LOGISTIC, LINEAR, true);

    // Neural network test 2: Just for fun
    // neuralNetworkFun();

    // Testing a single neuron/perceptron
    // std::cout << std::endl;
    perceptronTest();

    return 0;
}
