package main;
import neuralNetwork.NetworkFacade;
import neuralNetwork.NeuralNetwork;
import neuralNetwork.states.TrainingState;
import neuralNetwork.states.TestingState;
import data.AdapterHandler;

public class Main
{
    public static void main(String[] args)
    {
        // OPTION 1: Quick start with Facade (uses AdapterHandler internally)
        NetworkFacade NF = new NetworkFacade();
        NF.runFullExperiment();
        
        // OPTION 2: Manual control with different file types
        /*
        AdapterHandler reader = new AdapterHandler();
        int[] hidden = {128, 64};
        double learningRate = 0.01;
        NeuralNetwork nn = new NeuralNetwork(hidden, learningRate, reader);
        
        // Works with any file format - CSV, binary, or folders
        nn.setMode(new TrainingState());
        nn.processData("dataset/mnist/train");  // Can be CSV, .gz, or folder
        
        nn.setMode(new TestingState());
        nn.processData("dataset/mnist/test");
        */
    }
}
