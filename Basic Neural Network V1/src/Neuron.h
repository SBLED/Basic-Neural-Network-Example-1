#pragma once

#include "Connection.h"
#include <vector>
#include <cstdlib>

class Neuron; // Forward declaration

typedef std::vector<Neuron> Layer;

class Neuron {
private:
	static double eta; // Overall net learning rate: 0.0 - Slow to learn, 0.2 - medium learning, 1.0 - reckless learning
	static double alpha; // Multiplier of last weight change: momentum = a fraction of the previous delta weight. 0.0 - no momentum, 0.5 - moderate momentum. 

	double m_outputVal, m_gradient;
	unsigned m_myIndex;
	std::vector<Connection> m_outputWeights;

	static double transferFunction(double x); // Currently hyperbolic tangent function [-1.0, 1.0], could also use sigmoid or RELU.
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); } // Provides random number between 0 and 1
	const double sumDOW(const Layer& nextLayer);


public:

	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal() const{ return m_outputVal; }

	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);

};