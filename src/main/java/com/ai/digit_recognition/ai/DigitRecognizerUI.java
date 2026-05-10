package com.ai.digit_recognition.ai;
import java.io.InputStream;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import javax.swing.*;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;

public class DigitRecognizerUI extends JFrame {

    private BufferedImage canvasImage;

    private Graphics2D g2;

    private JLabel predictionLabel;

    private MultiLayerNetwork model;

    public DigitRecognizerUI() throws Exception {

        // Load trained CNN model
        InputStream modelStream =
        getClass().getClassLoader()
                .getResourceAsStream("mnist-model.zip");

        if (modelStream == null) {

        throw new RuntimeException(
                "Model file not found in resources!"
        );
        }

        model = ModelSerializer.restoreMultiLayerNetwork(
                modelStream
        );

        setTitle("Handwritten Digit Recognizer");

        setSize(700, 800);

        setLocationRelativeTo(null);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Main container
        JPanel mainPanel = new JPanel();

        mainPanel.setLayout(
                new BoxLayout(mainPanel, BoxLayout.Y_AXIS)
        );

        mainPanel.setBorder(
                BorderFactory.createEmptyBorder(
                        30,
                        30,
                        30,
                        30
                )
        );

        // Prediction Label
        predictionLabel = new JLabel(
                "Prediction: ",
                SwingConstants.CENTER
        );

        predictionLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        predictionLabel.setFont(
                new Font("Arial", Font.BOLD, 38)
        );

        predictionLabel.setBorder(
                BorderFactory.createEmptyBorder(
                        10,
                        10,
                        30,
                        10
                )
        );

        // Create drawing canvas
        canvasImage = new BufferedImage(
                420,
                420,
                BufferedImage.TYPE_BYTE_GRAY
        );

        g2 = canvasImage.createGraphics();

        g2.setRenderingHint(
                RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON
        );

        // Black background
        g2.setColor(Color.BLACK);

        g2.fillRect(0, 0, 420, 420);

        // White brush
        g2.setColor(Color.WHITE);

        g2.setStroke(
                new BasicStroke(
                        10,
                        BasicStroke.CAP_ROUND,
                        BasicStroke.JOIN_ROUND
                )
        );

        JPanel drawPanel = new JPanel() {

            @Override
            protected void paintComponent(Graphics g) {

                super.paintComponent(g);

                g.drawImage(canvasImage, 0, 0, null);
            }
        };

        drawPanel.setPreferredSize(
                new Dimension(420, 420)
        );

        drawPanel.setMaximumSize(
                new Dimension(420, 420)
        );

        drawPanel.setAlignmentX(Component.CENTER_ALIGNMENT);

        drawPanel.setBorder(
                BorderFactory.createLineBorder(
                        Color.DARK_GRAY,
                        2
                )
        );

        // Drawing logic
        drawPanel.addMouseMotionListener(
                new MouseMotionAdapter() {

                    @Override
                    public void mouseDragged(MouseEvent e) {

                        int x = e.getX();

                        int y = e.getY();

                        g2.fillOval(
                                x - 5,
                                y - 5,
                                10,
                                10
                        );

                        repaint();
                    }
                }
        );

        // Buttons
        JButton predictButton =
                new JButton("Predict");

        JButton clearButton =
                new JButton("Clear");

        // Button styling
        Font buttonFont =
                new Font("Arial", Font.BOLD, 20);

        predictButton.setFont(buttonFont);

        clearButton.setFont(buttonFont);

        predictButton.setPreferredSize(
                new Dimension(160, 55)
        );

        clearButton.setPreferredSize(
                new Dimension(160, 55)
        );

        // Predict action
        predictButton.addActionListener(e -> {

            try {

                int prediction = predictDigit();

                predictionLabel.setText(
                        "Prediction: " + prediction
                );

            } catch (Exception ex) {

                ex.printStackTrace();
            }
        });

        // Clear action
        clearButton.addActionListener(e -> {

            g2.setColor(Color.BLACK);

            g2.fillRect(0, 0, 420, 420);

            g2.setColor(Color.WHITE);

            predictionLabel.setText(
                    "Prediction: "
            );

            repaint();
        });

        JPanel buttonPanel = new JPanel();

        buttonPanel.setBorder(
                BorderFactory.createEmptyBorder(
                        30,
                        10,
                        10,
                        10
                )
        );

        buttonPanel.add(predictButton);

        buttonPanel.add(Box.createHorizontalStrut(20));

        buttonPanel.add(clearButton);

        // Add components
        mainPanel.add(predictionLabel);

        mainPanel.add(drawPanel);

        mainPanel.add(buttonPanel);

        add(mainPanel);

        setVisible(true);
    }

    // MNIST-style preprocessing
    private BufferedImage cropAndResize(
            BufferedImage original
    ) {

        int minX = original.getWidth();
        int minY = original.getHeight();

        int maxX = 0;
        int maxY = 0;

        // Find digit boundaries
        for (int y = 0; y < original.getHeight(); y++) {

            for (int x = 0; x < original.getWidth(); x++) {

                int pixel =
                        original.getRGB(x, y) & 0xFF;

                if (pixel > 20) {

                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);

                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        // Empty drawing protection
        if (maxX <= minX || maxY <= minY) {

            return original;
        }

        // Crop digit
        BufferedImage cropped =
                original.getSubimage(
                        minX,
                        minY,
                        maxX - minX + 1,
                        maxY - minY + 1
                );

        int croppedWidth = cropped.getWidth();
        int croppedHeight = cropped.getHeight();

        int newWidth;
        int newHeight;

        // Preserve aspect ratio
        if (croppedWidth > croppedHeight) {

            newWidth = 20;

            newHeight =
                    (int)((20.0 / croppedWidth)
                            * croppedHeight);

        } else {

            newHeight = 20;

            newWidth =
                    (int)((20.0 / croppedHeight)
                            * croppedWidth);
        }

        // Resize smoothly
        Image scaled =
                cropped.getScaledInstance(
                        newWidth,
                        newHeight,
                        Image.SCALE_SMOOTH
                );

        // Final 28x28 image
        BufferedImage centered =
                new BufferedImage(
                        28,
                        28,
                        BufferedImage.TYPE_BYTE_GRAY
                );

        Graphics2D g =
                centered.createGraphics();

        g.setColor(Color.BLACK);

        g.fillRect(0, 0, 28, 28);

        int x = (28 - newWidth) / 2;

        int y = (28 - newHeight) / 2;

        g.drawImage(
                scaled,
                x,
                y,
                null
        );

        g.dispose();

        return centered;
    }

    private int predictDigit() throws Exception {

        BufferedImage processed =
                cropAndResize(canvasImage);

        File tempFile =
                new File("temp.png");

        ImageIO.write(
                processed,
                "png",
                tempFile
        );

        NativeImageLoader loader =
                new NativeImageLoader(
                        28,
                        28,
                        1
                );

        INDArray image =
                loader.asMatrix(tempFile);

        // Normalize
        ImagePreProcessingScaler scaler =
                new ImagePreProcessingScaler(0, 1);

        scaler.transform(image);

        // CNN prediction
        INDArray output =
                model.output(image);

        return Nd4j.argMax(output, 1)
                .getInt(0);
    }

    public static void main(String[] args)
            throws Exception {

        new DigitRecognizerUI();
    }
}