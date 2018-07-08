#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <math.h>


// --------------------------------------- Neural Network Layer ---------------------------------------
NeuralNetworkLayer::NeuralNetworkLayer()
{
    isInputLayer = false;
}

NeuralNetworkLayer::NeuralNetworkLayer(int inputSize, int numberOfNeurons)
{
    setNumberOfNeurons(numberOfNeurons);
    setInputSize(inputSize);
    isInputLayer = false;
}

NeuralNetworkLayer::~NeuralNetworkLayer()
{

}

void NeuralNetworkLayer::setInputSize(int size)
{
    for (int i = 0; i < neurons.size(); i++)
        neurons[i].initWeightMatrix(size);

}

void NeuralNetworkLayer::setNumberOfNeurons(int numberOfNeurons)
{
    neurons.setSize(numberOfNeurons);
}

void NeuralNetworkLayer::setNextLayer(NeuralNetworkLayer& _nextLayer)
{
    nextLayer = &_nextLayer;
    nextLayer->setInputSize(neurons.size());
}

void NeuralNetworkLayer::setActivationFunction(EActivationFunction activationFunction)
{
    for (int i = 0; i < neurons.size(); i++)
        neurons[i].activationFunctionEnum = activationFunction;
}

void NeuralNetworkLayer::forwardPropagation(Matrix<float>& dataSample)
{
    if (isInputLayer)
    {
        // Input layer, pass the results directly
        results = dataSample;
        if (nextLayer != nullptr)
            nextLayer->forwardPropagation(results);
    }
    else
    {
        // Not input layer, perform prediction of data
        results.setSize(neurons.size(), 1);

        for (int i = 0; i < neurons.size(); i++)
            results[i][0] = neurons[i].predict(dataSample);

        if (nextLayer != nullptr)
            nextLayer->forwardPropagation(results);
    }
}

int NeuralNetworkLayer::size()
{
    return neurons.size();
}

float NeuralNetworkLayer::activationFunction(float input)
{
    if (neurons.size() > 0)
        return neurons[0].activationFunction(input);
    else std::cout << "FATAL ERROR: No neurons present in layer." << std::endl;
    exit(-1);
}

float NeuralNetworkLayer::derivedActivationFunction(float input)
{
    if (neurons.size() > 0)
        return neurons[0].derivedActivationFunction(input);
    else std::cout << "FATAL ERROR: No neurons present in layer." << std::endl;
    exit(-1);
}


// --------------------------------------- Neural Network ---------------------------------------

NeuralNetwork::NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, int numberOfHiddenLayers)
{
    learningRate = 0.5f;

    // Create the layers and set the appropriate parameters/settings
    layers.setSize(2 + numberOfHiddenLayers);
    for (int i = 0; i < layers.size(); i++)
    {
        // Set the number of neurons of the layer
        if (i == 0)
        {
            layers[i].setNumberOfNeurons(inputLayerSize);
            layers[i].isInputLayer = true;
            layers[i].setInputSize(inputLayerSize);
        }
        else if (i < layers.size() - 1)
            layers[i].setNumberOfNeurons(hiddenLayerSize);
        else
            layers[i].setNumberOfNeurons(outputLayerSize);

        // Link the current layer to the previous layer
        if (i != 0)
            layers[i - 1].setNextLayer(layers[i]);
    }
}

NeuralNetwork::~NeuralNetwork()
{

}

void NeuralNetwork::forwardPropagation(Matrix<float>& dataSample)
{
    layers[0].forwardPropagation(dataSample);
}

void NeuralNetwork::backpropagation(Matrix<float>& dataSample, Array<float>& classificationVector)
{
    /// First forward propagate
    forwardPropagation(dataSample);

    /// Second calculate delta value for each layer except the input layer
    // Using Matrix instead of Array to ease delta * weight calculation
    Array<Matrix<float>> delta(layers.size());

    // Calculate the delta values for the output layer
    // The number of delta per layer is equivalent to the number of neurons
    int outputLayer = layers.size() - 1;
    delta[outputLayer].setSize(layers[outputLayer].size(), 1);
    for (int outputNeuronID = 0; outputNeuronID < layers[outputLayer].size(); outputNeuronID++)
    {
        // (t - y) * derived_activation_function
        delta[outputLayer][outputNeuronID][0] = classificationVector[outputNeuronID] - layers[outputLayer].results[outputNeuronID][0]; // (t - y)
        delta[outputLayer][outputNeuronID][0] *= layers[outputLayer].derivedActivationFunction(layers[outputLayer].neurons[outputNeuronID].lastNetInput); // Derived value multiplied
    }

    // Calculate the delta values for all hidden layers
    // (Loop starts from the second last layer backwards)
    for (int x = layers.size() - 2; x >= 1; x--) // x >= 1 because ignore the input layer
    {
        // The number of delta per layer is equivalent to the number of neurons
        delta[x].setSize(layers[x].size(), 1);

        // Calculate the delta for each neuron
        for (int y = 0; y < layers[x].size(); y++)
        {
            delta[x][y][0] = 0;
            for (int z = 0; z < layers[x + 1].size(); z++)
                delta[x][y][0] += layers[x + 1].neurons[z].weightMatrix[y][0] * delta[x + 1][z][0]; // w * delta
            delta[x][y][0] *= layers[x].derivedActivationFunction(layers[x].neurons[y].lastNetInput); // Derived value multiplied
        }
    }

    /// Third update the weights for all hidden layers and the output layer
    // Loop for each layer
    for (int i = 1; i < layers.size(); i++)
    {
        // Loop for each neuron
        for (int j = 0; j < layers[i].size(); j++)
        {
            Matrix<float> &targetWeightMatrix = layers[i].neurons[j].weightMatrix;

            // Update bias: (learning_rate * 1 (bias) * delta_value)
            targetWeightMatrix[0][0] += learningRate * 1 * delta[i][j][0];

            // Update all else: (learning_rate * activation_value in layer i - 1 * delta_value)
            for (int k = 1; k < targetWeightMatrix.getSizeX(); k++)
                targetWeightMatrix[k][0] += learningRate * layers[i - 1].results[k - 1][0] * delta[i][j][0];
        }
    }
}

void NeuralNetwork::backpropagationStochastic(Array<Matrix<float>>& dataSamples, Array<Array<float>>& classificationVectors, int epochs)
{
    if (dataSamples.size() != classificationVectors.size())
    {
        std::cout << "Error: Input data sample vector size is not equal to the classification vector size!" << std::endl;
        return;
    }

    // Learn the samples epochs times
    for (int x = 0; x < epochs; x++)
    {
        // Access all samples stochastically
        std::vector<int> indicesLeft;
        for (int i = 0; i < dataSamples.size(); i++)
            indicesLeft.push_back(i);

        while (indicesLeft.size() != 0)
        {
            int randomAccessIndex = rand() % indicesLeft.size();
            backpropagation(dataSamples[randomAccessIndex], classificationVectors[randomAccessIndex]);
            indicesLeft.erase(indicesLeft.begin() + randomAccessIndex);
        }
    }
}

/**
    Get and return the ouput neuron ID with the largest response
    This represents the class respectively
*/
int NeuralNetwork::getClassWithMaxResponse()
{
    NeuralNetworkLayer &targetLayer = layers[layers.size() - 1];
    float maxResponse = targetLayer.results[0][0];
    int neuronID = 0;
    for (int i = 1; i < targetLayer.results.getSizeX(); i++)
    {
        if (targetLayer.results[i][0] > maxResponse)
        {
            maxResponse = targetLayer.results[i][0];
            neuronID = i;
        }
    }
    return neuronID;
}

/**
    Opposite to above
*/
int NeuralNetwork::getClassWithMinResponse()
{
    NeuralNetworkLayer &targetLayer = layers[layers.size() - 1];
    float minResponse = targetLayer.results[0][0];
    int neuronID = 0;
    for (int i = 1; i < targetLayer.results.getSizeX(); i++)
    {
        if (targetLayer.results[i][0] < minResponse)
        {
            minResponse = targetLayer.results[i][0];
            neuronID = i;
        }
    }
    return neuronID;
}

/**
    Get the largest response the return it
*/
float NeuralNetwork::getMaxResponse()
{
    NeuralNetworkLayer &targetLayer = layers[layers.size() - 1];
    float maxResponse = targetLayer.results[0][0];
    for (int i = 1; i < targetLayer.results.getSizeX(); i++)
    {
        if (targetLayer.results[i][0] > maxResponse)
            maxResponse = targetLayer.results[i][0];
    }
    return maxResponse;
}

/**
    Opposite to the above
*/
float NeuralNetwork::getMinResponse()
{
    NeuralNetworkLayer &targetLayer = layers[layers.size() - 1];
    float minResponse = targetLayer.results[0][0];
    for (int i = 1; i < targetLayer.results.getSizeX(); i++)
    {
        if (targetLayer.results[i][0] < minResponse)
            minResponse = targetLayer.results[i][0];
    }
    return minResponse;
}

int NeuralNetwork::getNegatedMaxResponse()
{
    switch ((int) round(getMaxResponse()))
    {
        case 0: return 1;
        case 1: return 0;
        default: return -1;
    }
}

int NeuralNetwork::getNegatedMinResponse()
{
    switch ((int) round(getMinResponse()))
    {
        case 0: return 1;
        case 1: return 0;
        default: return -1;
    }
}
