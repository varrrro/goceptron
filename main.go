package main

import (
	"log"

	"github.com/varrrro/goceptron/internal/model"
	"github.com/varrrro/goceptron/internal/util"
	"gonum.org/v1/gonum/mat"
)

const (
	mnistURL    = "http://yann.lecun.com/exdb/mnist/"
	localPath   = "data/mnist/"
	trainImages = "train-images-idx3-ubyte.gz"
	trainLabels = "train-labels-idx1-ubyte.gz"
	testImages  = "t10k-images-idx3-ubyte.gz"
	testLabels  = "t10k-labels-idx1-ubyte.gz"
)

func main() {
	/*
		err := util.DownloadMNIST(mnistURL, localPath, trainImages, trainLabels, testImages, testLabels)
		if err != nil {
			log.Println(err.Error())
			return
		}
	*/

	rawTrainImgs, trainLbls, err := util.ReadData(localPath, trainImages, trainLabels)
	if err != nil {
		log.Println(err.Error())
		return
	}

	rawTestImgs, testLbls, err := util.ReadData(localPath, testImages, testLabels)
	if err != nil {
		log.Println(err.Error())
		return
	}

	trainImgs := util.NormalizeImages(&rawTrainImgs)
	testImgs := util.NormalizeImages(&rawTestImgs)

	initPerceptron(&trainImgs, &testImgs, &trainLbls, &testLbls)
}

func initPerceptron(trainImgs, testImgs *[]mat.Vector, trainLabels, testLabels *[]int) {
	// Initialize weight and bias arrays
	weights := util.RandomWeights(784, 10)
	biases := util.RandomBiases(10)

	// Create the perceptron
	p := model.CreatePerceptron(784, 10, weights, biases)

	// Train the perceptron with the training set for a number of epochs
	p.Train(*trainImgs, *trainLabels, 10)

	// Test the perceptron with the test set
	p.Test(*testImgs, *testLabels)
}
