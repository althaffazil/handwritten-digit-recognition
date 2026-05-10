package com.ai.digit_recognition.ai;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.deeplearning4j.nn.conf.inputs.InputType;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TrainModel {

    public static void main(String[] args)
            throws Exception {

        int batchSize = 64;

        int outputNum = 10;

        int epochs = 2;

        // Load MNIST
        var mnistTrain =
                new MnistDataSetIterator(
                        batchSize,
                        true,
                        12345
                );

        var mnistTest =
                new MnistDataSetIterator(
                        batchSize,
                        false,
                        12345
                );

        // CNN Configuration
        MultiLayerConfiguration config =
                new NeuralNetConfiguration.Builder()

                        .updater(new Adam(0.001))

                        .list()

                        // Convolution Layer
                        .layer(
                                new ConvolutionLayer.Builder(5, 5)

                                        .nIn(1)

                                        .stride(1, 1)

                                        .nOut(20)

                                        .activation(Activation.RELU)

                                        .build()
                        )

                        // Pooling Layer
                        .layer(
                                new SubsamplingLayer.Builder(
                                        SubsamplingLayer.PoolingType.MAX
                                )

                                        .kernelSize(2, 2)

                                        .stride(2, 2)

                                        .build()
                        )

                        // Dense Layer
                        .layer(
                                new DenseLayer.Builder()

                                        .nOut(128)

                                        .activation(Activation.RELU)

                                        .build()
                        )

                        // Output Layer
                        .layer(
                                new OutputLayer.Builder(
                                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD
                                )

                                        .nOut(outputNum)

                                        .activation(Activation.SOFTMAX)

                                        .build()
                        )

                        // CNN Input Shape
                        .setInputType(
                                InputType.convolutionalFlat(
                                        28,
                                        28,
                                        1
                                )
                        )

                        .build();

        // Create model
        MultiLayerNetwork model =
                new MultiLayerNetwork(config);

        model.init();

        model.setListeners(
                new ScoreIterationListener(100)
        );

        System.out.println("CNN Training started...");

        // Train
        for (int i = 0; i < epochs; i++) {

            model.fit(mnistTrain);

            System.out.println(
                    "Epoch " + (i + 1) + " complete"
            );
        }

        System.out.println("Training complete.");

        // Evaluate
        Evaluation eval =
                model.evaluate(mnistTest);

        System.out.println(eval.stats());

        // Save model
        ModelSerializer.writeModel(
                model,
                "mnist-model.zip",
                true
        );

        System.out.println(
                "CNN model saved as mnist-model.zip"
        );
    }
}