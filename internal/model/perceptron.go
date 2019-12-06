package model

import (
	"log"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Perceptron for input classification.
type Perceptron struct {
	inputSize  int
	outputSize int
	input      mat.Vector
	output     mat.Vector
	weights    []mat.Vector
	biases     mat.Vector
}

// CreatePerceptron with the given parameters.
func CreatePerceptron(nInputs, nOutputs int, weights [][]float64, biases []float64) *Perceptron {
	w := make([]mat.Vector, nOutputs)
	for i := 0; i < nOutputs; i++ {
		w[i] = mat.NewVecDense(nInputs, weights[i])
	}

	b := mat.NewVecDense(nOutputs, biases)

	return &Perceptron{
		inputSize:  nInputs,
		outputSize: nOutputs,
		weights:    w,
		biases:     b,
	}
}

// FeedForward an input through the perceptron.
func (p *Perceptron) FeedForward(pattern mat.Vector) {
	p.input = pattern

	sumVec := mat.NewVecDense(p.inputSize, nil)
	y := mat.NewVecDense(p.outputSize, nil)
	for i := 0; i < p.outputSize; i++ {
		sumVec.MulElemVec(p.input, p.weights[i])

		z := mat.Sum(sumVec)
		z += p.biases.AtVec(i)

		y.SetVec(i, activation(z))
	}

	p.output = y
}

func activation(z float64) float64 {
	if z >= 0 {
		return 1.0
	} else {
		return 0.0
	}
}

// UpdateWeights of the perceptron based on the target output.
func (p *Perceptron) UpdateWeights(t int) {
	newWeights := mat.NewVecDense(p.inputSize, nil)
	newBiases := mat.NewVecDense(p.outputSize, nil)

	for i := 0; i < p.outputSize; i++ {
		if i == t && p.output.AtVec(i) != 1.0 {
			newWeights.AddVec(p.weights[i], p.input)
			newBiases.SetVec(i, p.biases.AtVec(i)+1.0)
		} else if i != t && p.output.AtVec(i) != 0.0 {
			newWeights.SubVec(p.weights[i], p.input)
			newBiases.SetVec(i, p.biases.AtVec(i)-1.0)
		} else {
			continue
		}

		p.weights[i] = mat.VecDenseCopyOf(newWeights)
	}

	p.biases = newBiases
}

// Train the network with the given patterns.
func (p *Perceptron) Train(patterns []mat.Vector, labels []int, epochs int) {
	log.Println(">> Training started")
	start := time.Now()

	for i := 0; i < epochs; i++ {
		log.Printf("> Starting epoch %d", i)

		for j, pattern := range patterns {

			p.FeedForward(pattern)
			p.UpdateWeights(labels[j])
		}

		p.Test(patterns, labels)
	}

	elapsed := time.Since(start)
	log.Printf(">> Training finished | Time elapsed: %f", elapsed.Seconds())
}

// Test the perceptron with the given patterns.
func (p *Perceptron) Test(patterns []mat.Vector, labels []int) {
	failures := 0

	for i, pattern := range patterns {
		p.FeedForward(pattern)

		if p.output.AtVec(labels[i]) != 1.0 {
			failures++
		}
	}

	failureRate := (float64(failures) / float64(len(patterns))) * float64(100)

	log.Println(">>>> Test finished")
	log.Printf("Number of patterns: %d", len(patterns))
	log.Printf("Number of failures: %d", failures)
	log.Printf("Falure rate: %f", failureRate)
}
