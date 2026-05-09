package main;

import neuralNetwork.states.*;
import neuralNetwork.NeuralNetwork;

public class Main
{

    public static void main( String[] args )
    {
        double LEARNING_RATE = 0.01;

        int[] hidden = { 128, 64 };

        String trainingFilePath = "dataset/mnist_train.csv";
        String testingFilePath = "dataset/mnist_test.csv";

        NeuralNetwork newNet = new NeuralNetwork( hidden, LEARNING_RATE );

        // 2. PHASE 1: Training
        // MOVED FROM: newNet.train() 
        // We now explicitly define the mode

        newNet.setMode(new TrainingState());
        newNet.processData(trainingFilePath);

        // 3. PHASE 2: Testing
        // MOVED FROM: newNet.test()
        // Switching state changes the internal behavior of processData()
        
        newNet.setMode(new TestingState());
        newNet.processData(testingFilePath);
        
    }

    

}
