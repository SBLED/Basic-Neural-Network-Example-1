#pragma once

#include "Neuron.h"
//#include <cmath>

double Neuron::eta = 0.15; // Overall net learning rate: 0.0 - Slow to learn, 0.2 - medium learning, 1.0 - reckless learning [0.0, 1]
double Neuron::alpha; // Multiplier of last weight change: momentum = a fraction of the previous delta weight. 0.0 - no momentum, 0.5 - moderate momentum. [0.0, n]

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {

	m_myIndex = myIndex;

	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

}

void Neuron::feedForward(const Layer& prevLayer) {

	double sum = 0.0;

	// Sum the previous layer's outputs (which are inputs)
	// Include the bias node from the previous layer.

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);

}

double Neuron::transferFunction(double x) {
	// Can use any transfer function (e.g. Sigmoid, RELU). Just set up training data to scale output so that output values are always in correct range for transfer function
	return tanh(x); //tanh output range: [-1.0, 1.0]
}

double Neuron::transferFunctionDerivative(double x) {
	
	return 1.0 - x * x; //tanh derivative (quick approximation that works over this interval)
}

void Neuron::calcOutputGradients(double targetVal) {
	// Pushes the Net to reduce overall error using the mean squared errors. For cross entropy, just use m_gradient = targetVal - m_outputVal;
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer) {

	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);

}

const double Neuron::sumDOW(const Layer& nextLayer) {

	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::updateInputWeights(Layer& prevLayer) {

	// The weights to be updated are in the Connection container in the preceding layer's neurons.
	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron& neuron = prevLayer[n]; // neuron is the previous layer's neuron that is being updated.
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			//Individual input, magnified by the gradient and train rate
			eta * neuron.getOutputVal() * m_gradient // eta (overall net learning rate): 0.0 - Slow to learn, 0.2 - medium learning, 1.0 - reckless learning
			+ alpha * oldDeltaWeight; //Also add "alpha" momentum = a fraction of the previous delta weight. 0.0 - no momentum, 0.5 - moderate momentum

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;

	}

}
