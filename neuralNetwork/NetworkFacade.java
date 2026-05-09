package neuralNetwork;

import neuralNetwork.states.*;

public class NetworkFacade {
    private NeuralNetwork network;
    private NetworkState ns;
    // Armaghan's State pattern will go here 

    public NetworkFacade() {

        int[] hiddenLayers = {128, 64};
        double learningRate = 0.01;
        this.network = new NeuralNetwork(hiddenLayers, learningRate);
    }

    public void runFullExperiment() {
        
        System.out.println("Starting Setup...");

        String trainingFilePath = "dataset/mnist_train.csv";
        String testingFilePath = "dataset/mnist_test.csv";
        
        // Armaghan's state logic will just be a method call here:
        // stateManager.setTrainingState(); 
        ns = new TrainingState();
        network.setMode(ns);
        network.processData(trainingFilePath);
        
        // stateManager.setTestingState();
        ns = new TestingState();
        network.setMode(ns);
        network.processData(testingFilePath);
        
        System.out.println("Experiment Complete.");
    }
}