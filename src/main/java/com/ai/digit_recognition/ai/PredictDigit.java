package com.ai.digit_recognition.ai;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class PredictDigit {

    public static void main(String[] args) throws Exception {

        // Load trained model
        MultiLayerNetwork model =
                ModelSerializer.restoreMultiLayerNetwork(
                        "mnist-model.zip"
                );

        System.out.println("Model loaded successfully.");

        // Load image
        File imageFile = new File("images/test.jpg");

        NativeImageLoader loader =
                new NativeImageLoader(28, 28, 1);

        INDArray image = loader.asMatrix(imageFile);

        // Normalize pixel values
        ImagePreProcessingScaler scaler =
                new ImagePreProcessingScaler(0, 1);

        scaler.transform(image);

        // Flatten image for dense network
        

        // Predict
        INDArray output = model.output(image);

        int predictedDigit =
                Nd4j.argMax(output, 1).getInt(0);

        System.out.println("Predicted Digit: " + predictedDigit);

        System.out.println("Prediction Probabilities:");
        System.out.println(output);
    }
}