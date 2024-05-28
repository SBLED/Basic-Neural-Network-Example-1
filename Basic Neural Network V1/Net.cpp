#pragma once

#include "Net.h"
#include <iostream>
#include <cassert>

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over.

Net::Net(const std::vector<unsigned>& topology) {
	unsigned numLayers = topology.size();

	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer()); // Creates a new layer

		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1]; // If layerNum is output layer (highest layerNum), then outputs = 0; Else, numOutputs = 

		// Loops to fill new layer with ith neurons and adds a bias neuron to the layer
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) { // neuronNum <= topology[...] allows loop to iterate one additional time, allowing for the addition of a bias neuron
			m_layers.back().push_back(Neuron(numOutputs, neuronNum)); // Adds a neuron to the newly created layer

			std::cout << "Created a Neuron." << std::endl;
		}

		//Force bias neuron's output value to 1.0. Bias neuron is the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}

}

void Net::feedForward(const std::vector<double>& inputVals) {

	assert(inputVals.size() == m_layers[0].size() - 1);

	//Assign (latch) the input val into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	//Forward propogation
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1]; // Creating a pointer to the previous layer

		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}

}

void Net::backProp(const std::vector<double>& targetVals) {

	// Calculate overall net error of entire net (RMS of output neuron errors) (Tells us if training is successful)
	Layer& outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}

	m_error /= outputLayer.size() - 1; // Get average error squared
	m_error = sqrt(m_error); // RMS

	// Implements a recent average measurement. Not used for net itself, but aids in displaying net performance to user.
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

	//Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//Calculate gradients on hidden layers
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer& hiddenLayer = m_layers[layerNum];
		Layer& nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}

	}


	//For all layers from outputs to first hidden layer, update connection weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}

}


void Net::getResults(std::vector<double>& resultVals) const {

	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}

}