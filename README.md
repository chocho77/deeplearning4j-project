# DeepLearning4j Примерен Проект

## Описание
Този проект демонстрира използването на Deeplearning4j за създаване на дълбоки невронни мрежи.

## Изисквания
- Java 11 или по-висока
- Maven 3.6+

## Инсталация
```bash
git clone <repository-url>
cd deeplearning4j-project
mvn clean install

## Execute commands
```bash
# Compile
mvn compile

#Tests
mvn test

# Package
mvn package

# Start on MNIST classifiler
mvn exec:java -Dexec.mainClass="com.example.MNISTClassifiler"

# Start on ND4J example
mvn exec:java -Dexec.mainClass="com.example.ND4JExample"