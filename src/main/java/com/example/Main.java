package com.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);
    
    public static void main(String[] args) {
        log.info("Стартиране на DeepLearning4j пример...");
        
        // Създаване на проста XOR мрежа
        createAndTrainXORNetwork();
    }
    
    public static void createAndTrainXORNetwork() {
        log.info("Създаване на XOR невронна мрежа");
        
        // Подготовка на XOR данни
        INDArray input = Nd4j.create(new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        });
        
        INDArray output = Nd4j.create(new double[][]{
            {1, 0}, // 0 XOR 0 = 0
            {0, 1}, // 0 XOR 1 = 1
            {0, 1}, // 1 XOR 0 = 1
            {1, 0}  // 1 XOR 1 = 0
        });
        
        // Конфигурация на мрежата
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(4)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(4)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        log.info("Модел създаден успешно");
        log.info(model.summary());
        
        // Трениране
        log.info("Започване на тренировка...");
        for (int i = 0; i < 1000; i++) {
            model.fit(input, output);
            if (i % 200 == 0) {
                log.info("Епоха {}: грешка = {}", i, model.score());
            }
        }
        
        // Тестване
        log.info("Тестване на модела:");
        INDArray predictions = model.output(input);
        
        for (int i = 0; i < 4; i++) {
            double x1 = input.getDouble(i, 0);
            double x2 = input.getDouble(i, 1);
            double pred0 = predictions.getDouble(i, 0);
            double pred1 = predictions.getDouble(i, 1);
            int predictedClass = pred0 > pred1 ? 0 : 1;
            int expectedClass = (i == 0 || i == 3) ? 0 : 1;
            
            log.info("XOR({}, {}) = {}, очаквано: {} {}", 
                (int)x1, (int)x2, predictedClass, expectedClass,
                predictedClass == expectedClass ? "✓" : "✗");
        }
    }
}