#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Array.h"
#include "Matrix.h"
#include "Neuron.h"

class NeuralNetworkLayer
{
    public:
        NeuralNetworkLayer();
        NeuralNetworkLayer(int inputSize, int numberOfNeurons);
        virtual ~NeuralNetworkLayer();

        void setInputSize(int size);
        void setNumberOfNeurons(int numberOfNeurons);
        void setActivationFunction(EActivationFunction activationFunction);
        void setNextLayer(NeuralNetworkLayer& _nextLayer);
        void forwardPropagation(Matrix<float> &dataSample);

        float outputValue(Matrix<float>& dataSample, int neuronID);
        float activationFunction(float input);
        float derivedActivationFunction(float input);

        int size();

        bool isInputLayer;
        Matrix<float> results;
        Array<Neuron> neurons;

    private:
        NeuralNetworkLayer* nextLayer = nullptr;
};

/**
    Creating this neural network for online learning
*/
class NeuralNetwork
{
    public:
        NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, int numberOfHiddenLayers);
        virtual ~NeuralNetwork();

        // Neural network related functions
        void forwardPropagation(Matrix<float> &dataSample);
        void backpropagation(Matrix<float> &dataSample, Array<float> &classificationVector);
        void backpropagationStochastic(Array<Matrix<float>> &dataSamples, Array<Array<float>> &classificationVectors, int epochs);

        // Functions for retrieving the result calculated
        int getClassWithMaxResponse();
        int getClassWithMinResponse();
        float getMaxResponse();
        float getMinResponse();

        // Functions for negated response values (only works for values within range of 0 and 1
        int getNegatedMaxResponse();
        int getNegatedMinResponse();

        Array<NeuralNetworkLayer> layers;
        float learningRate;

    protected:

    private:
};

#endif // NEURALNETWORK_H
