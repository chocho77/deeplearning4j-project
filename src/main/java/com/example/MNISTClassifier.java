package com.example;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Примерен клас за класификация на MNIST ръкописни цифри с Deeplearning4j
 */
public class MNISTClassifier {
    
    private static final Logger log = LoggerFactory.getLogger(MNISTClassifier.class);
    
    // Параметри на модела
    private static final int SEED = 123;
    private static final int EPOCHS = 10;
    private static final int BATCH_SIZE = 64;
    private static final int NUM_CLASSES = 10;
    private static final int NUM_INPUTS = 28 * 28; // 784 пиксела

    public static void main(String[] args) throws IOException {
        
        log.info("Зареждане на MNIST данни...");
        
        // Зареждане на тренировъчни и тестови данни
        DataSetIterator mnistTrain = new MnistDataSetIterator(BATCH_SIZE, true, SEED);
        DataSetIterator mnistTest = new MnistDataSetIterator(BATCH_SIZE, false, SEED);
        
        log.info("Създаване на конфигурация на модела...");
        
        // Конфигурация на невронната мрежа
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.01, 0.9))
                .l2(1e-4)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(NUM_INPUTS)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(256)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(128)
                        .nOut(NUM_CLASSES)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        
        // Създаване на модела
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        // Добавяне на listener за проследяване на тренировката
        model.setListeners(new ScoreIterationListener(100));
        
        log.info("Започване на тренировка...");
        
        // Трениране на модела
        for (int i = 0; i < EPOCHS; i++) {
            model.fit(mnistTrain);
            log.info("Приключен епох {}", i + 1);
            
            // Оценка на модела след всеки епох
            Evaluation eval = model.evaluate(mnistTest);
            log.info("Точност след епох {}: {}", i + 1, eval.accuracy());
            log.info("Статистика: \n{}", eval.stats());
            
            // Нулиране на итератора за следващия епох
            mnistTrain.reset();
            mnistTest.reset();
        }
        
        log.info("Тренировката приключи успешно!");
        
        // Финален резултат
        Evaluation finalEval = model.evaluate(mnistTest);
        log.info("Финален резултат:");
        log.info("Точност: {}", finalEval.accuracy());
        log.info("Прецизност: {}", finalEval.precision());
        log.info("Отзивчивост: {}", finalEval.recall());
        log.info("F1 резултат: {}", finalEval.f1());
        
        // Запазване на модела (опционално)
        // File locationToSave = new File("mnist-model.zip");
        // ModelSerializer.writeModel(model, locationToSave, true);
    }
}
