package com.example;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Визуализация на решенията на невронната мрежа
 */
public class NetworkVisualizer {
    
    public static void visualizeDecisionBoundary(MultiLayerNetwork model, int resolution) {
        System.out.println("\nВизуализация на решението на мрежата:");
        System.out.println("(0 = false, 1 = true)");
        System.out.println("=======================");
        
        for (int i = 0; i <= resolution; i++) {
            for (int j = 0; j <= resolution; j++) {
                double x = (double) i / resolution;
                double y = (double) j / resolution;
                
                INDArray input = Nd4j.create(new double[][]{{x, y}});
                INDArray output = model.output(input);
                
                int prediction = output.getDouble(0) > output.getDouble(1) ? 0 : 1;
                
                if (prediction == 0) {
                    System.out.print("0 ");
                } else {
                    System.out.print("1 ");
                }
            }
            System.out.println();
        }
    }
}