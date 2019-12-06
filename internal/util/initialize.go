package util

import (
	"math/rand"
)

// RandomWeights in [-1, 1].
func RandomWeights(inputs, neurons int) [][]float64 {
	weights := make([][]float64, neurons)

	for i := 0; i < neurons; i++ {
		weights[i] = make([]float64, inputs)

		for j := 0; j < inputs; j++ {
			weights[i][j] = (rand.Float64() * 2) - 1
		}
	}

	return weights
}

// RandomBiases in [-1, 1].
func RandomBiases(neurons int) []float64 {
	biases := make([]float64, neurons)

	for i := 0; i < neurons; i++ {
		biases[i] = (rand.Float64() * 2) - 1
	}

	return biases
}
