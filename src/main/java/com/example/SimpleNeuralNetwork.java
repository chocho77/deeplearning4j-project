package com.example;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Пример за невронна мрежа с 2 входа, 1 скрит слой и 2 изхода
 * Решава проблема XOR (изключващо или)
 */
public class SimpleNeuralNetwork {
    
    private static final Logger log = LoggerFactory.getLogger(SimpleNeuralNetwork.class);
    
    public static void main(String[] args) {
        
        // 1. ПОДГОТОВКА НА ДАННИТЕ
        // Входове: 4 примера с по 2 features
        INDArray input = Nd4j.create(new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        });
        
        // Изходи: 4 примера с по 2 изхода (one-hot encoding)
        // [1,0] = false/0, [0,1] = true/1
        INDArray output = Nd4j.create(new double[][]{
            {1, 0}, // 0 XOR 0 = 0
            {0, 1}, // 0 XOR 1 = 1
            {0, 1}, // 1 XOR 0 = 1
            {1, 0}  // 1 XOR 1 = 0
        });
        
        DataSet dataSet = new DataSet(input, output);
        
        log.info("Данни за тренировка:");
        log.info("Входове: \n{}", input);
        log.info("Изходи: \n{}", output);
        
        // 2. КОНФИГУРАЦИЯ НА МРЕЖАТА
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)                // За възпроизводимост
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.1))      // Learning rate
                .weightInit(WeightInit.XAVIER)
                .list()
                // Скрит слой - 2 входа, 4 неврона в скрития слой
                .layer(0, new DenseLayer.Builder()
                        .nIn(2)              // 2 входа
                        .nOut(4)              // 4 неврона в скрития слой
                        .activation(Activation.SIGMOID)
                        .name("hidden_layer")
                        .build())
                // Изходен слой - 4 входа от скрития слой, 2 изхода
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(4)               // Връзка със скрития слой
                        .nOut(2)               // 2 изхода
                        .activation(Activation.SOFTMAX)
                        .name("output_layer")
                        .build())
                .build();
        
        // 3. СЪЗДАВАНЕ НА МОДЕЛА
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        // Добавяне на listener за проследяване на грешката
        model.setListeners(new ScoreIterationListener(100));
        
        log.info("Архитектура на модела:");
        log.info(model.summary());
        
        log.info("Започване на тренировка...");
        
        // 4. ТРЕНИРАНЕ НА МРЕЖАТА
        int epochs = 1000;
        for (int i = 0; i < epochs; i++) {
            model.fit(dataSet);
            
            // Извеждане на напредък на всеки 100 епохи
            if ((i + 1) % 100 == 0) {
                log.info("Епоха {} от {}, грешка: {}", i + 1, epochs, model.score());
            }
        }
        
        // 5. ТЕСТВАНЕ НА МРЕЖАТА
        log.info("Тестване на обучената мрежа:");
        
        INDArray testInput = Nd4j.create(new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        });
        
        INDArray predictions = model.output(testInput);
        
        log.info("Предвиждания на мрежата:");
        for (int i = 0; i < 4; i++) {
            double[] pred = predictions.getRow(i).toDoubleVector();
            double[] expected = output.getRow(i).toDoubleVector();
            
            // Определяне на класа (0 или 1) от вероятностите
            int predictedClass = pred[0] > pred[1] ? 0 : 1;
            int expectedClass = expected[0] > expected[1] ? 0 : 1;
            int xorResult = expectedClass == 0 ? 0 : 1;
            
            log.info("Вход: [{}, {}] -> Очаквано: {} (XOR={}), Предвидено: {} ({}), Sig: [{:.4f}, {:.4f}]",
                    (int)testInput.getDouble(i, 0),
                    (int)testInput.getDouble(i, 1),
                    expectedClass == 0 ? "false/0" : "true/1",
                    xorResult,
                    predictedClass == 0 ? "false/0" : "true/1",
                    predictedClass == expectedClass ? "✓" : "✗",
                    pred[0], pred[1]);
        }
        
        // 6. ИЗВЛИЧАНЕ НА ТЕГЛА
        log.info("Анализ на теглата:");
        
        // Тегла на скрития слой
        INDArray hiddenWeights = model.getLayer(0).getParam("W");
        INDArray hiddenBias = model.getLayer(0).getParam("b");
        
        log.info("Тегла на скрития слой (2x4):\n{}", hiddenWeights);
        log.info("Bias на скрития слой (1x4):\n{}", hiddenBias);
        
        // Тегла на изходния слой
        INDArray outputWeights = model.getLayer(1).getParam("W");
        INDArray outputBias = model.getLayer(1).getParam("b");
        
        log.info("Тегла на изходния слой (4x2):\n{}", outputWeights);
        log.info("Bias на изходния слой (1x2):\n{}", outputBias);
    }
}
