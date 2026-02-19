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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Гъвкава имплементация на невронна мрежа с 2 входа, скрит слой и 2 изхода
 */
public class FlexibleNeuralNetwork {
    
    private MultiLayerNetwork model;
    private int hiddenNeurons;
    
    /**
     * Създава невронна мрежа с 2 входа, скрит слой и 2 изхода
     * @param hiddenNeurons брой неврони в скрития слой
     * @param learningRate скорост на обучение
     * @param activation функция на активация за скрития слой
     */
    public FlexibleNeuralNetwork(int hiddenNeurons, double learningRate, Activation activation) {
        this.hiddenNeurons = hiddenNeurons;
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(hiddenNeurons)
                        .activation(activation)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(hiddenNeurons)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        
        model = new MultiLayerNetwork(conf);
        model.init();
    }
    
    /**
     * Тренира мрежата върху XOR данни
     */
    public void train(int epochs) {
        // XOR данни
        INDArray input = Nd4j.create(new double[][]{
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        });
        
        INDArray output = Nd4j.create(new double[][]{
            {1, 0}, {0, 1}, {0, 1}, {1, 0}
        });
        
        System.out.println("Трениране на мрежа с " + hiddenNeurons + " неврона в скрития слой...");
        
        for (int i = 0; i < epochs; i++) {
            model.fit(input, output);
        }
    }
    
    /**
     * Правене на предвиждане
     */
    public int predict(double x1, double x2) {
        INDArray input = Nd4j.create(new double[][]{{x1, x2}});
        INDArray output = model.output(input);
        
        // Връща 0 или 1 в зависимост кой изход има по-висока стойност
        return output.getDouble(0) > output.getDouble(1) ? 0 : 1;
    }
    
    /**
     * Тестване на всички XOR комбинации
     */
    public void test() {
        System.out.println("\nТестване на мрежата:");
        System.out.println("x1\tx2\texpected\tpredicted\tcorrect");
        System.out.println("----------------------------------------");
        
        double[][] tests = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] expected = {0, 1, 1, 0};
        
        int correct = 0;
        for (int i = 0; i < tests.length; i++) {
            int pred = predict(tests[i][0], tests[i][1]);
            boolean isCorrect = (pred == expected[i]);
            if (isCorrect) correct++;
            
            System.out.printf("%.0f\t%.0f\t%d\t\t%d\t\t%s\n",
                tests[i][0], tests[i][1], expected[i], pred,
                isCorrect ? "✓" : "✗");
        }
        
        System.out.printf("\nТочност: %d/4 (%.0f%%)\n", correct, (correct/4.0)*100);
    }
    
    public static void main(String[] args) {
        // Тест с различен брой неврони в скрития слой
        System.out.println("=== ТЕСТ 1: 3 неврона в скрития слой ===");
        FlexibleNeuralNetwork net1 = new FlexibleNeuralNetwork(3, 0.1, Activation.SIGMOID);
        net1.train(1000);
        net1.test();
        
        System.out.println("\n=== ТЕСТ 2: 5 неврона в скрития слой ===");
        FlexibleNeuralNetwork net2 = new FlexibleNeuralNetwork(5, 0.1, Activation.RELU);
        net2.train(1000);
        net2.test();
        
        System.out.println("\n=== ТЕСТ 3: 10 неврона в скрития слой ===");
        FlexibleNeuralNetwork net3 = new FlexibleNeuralNetwork(10, 0.05, Activation.TANH);
        net3.train(1000);
        net3.test();
    }
}