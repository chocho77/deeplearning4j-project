package com.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Пример за работа с ND4J матрици
 */
public class ND4JExample {
    
    public static void main(String[] args) {
        
        // Създаване на матрици
        INDArray matrix1 = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray matrix2 = Nd4j.create(new double[]{5, 6, 7, 8}, new int[]{2, 2});
        
        System.out.println("Матрица 1:");
        System.out.println(matrix1);
        
        System.out.println("\nМатрица 2:");
        System.out.println(matrix2);
        
        // Математически операции
        System.out.println("\nСъбиране:");
        System.out.println(matrix1.add(matrix2));
        
        System.out.println("\nУмножение:");
        System.out.println(matrix1.mmul(matrix2));
        
        // Сигмоид функция
        System.out.println("\nСигмоид на матрица 1:");
        System.out.println(Transforms.sigmoid(matrix1));
        
        // Нормализация
        INDArray normalized = matrix1.sub(matrix1.mean()).div(matrix1.std());
        System.out.println("\nНормализирана матрица 1:");
        System.out.println(normalized);
    }
}
