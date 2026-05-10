package neuralNetwork;

import data.AdapterHandler;
import data.ReaderInterface;
import neuralNetwork.states.*;

public class NetworkFacade {
    private NeuralNetwork network;
    private NetworkState ns;
    private ReaderInterface reader;  // Reader for flexible file handling

    public NetworkFacade() {
        int[] hiddenLayers = {128, 64};
        double learningRate = 0.01;
        
        // Create adapter handler for flexible file format support
        this.reader = new AdapterHandler();
        
        // Pass reader to network constructor
        this.network = new NeuralNetwork(hiddenLayers, learningRate, reader);
    }

    public void runFullExperiment() {
        System.out.println("Starting Setup...");

        // These paths now support CSV, binary, or mixed folders
        String trainingFilePathCSV = "dataset/mnist_train.csv";
        String trainingFilePathBin = "dataset/train-labels-idx1-ubyte.gz";

        String testingFilePathBin = "dataset/t10k-labels-idx1-ubyte.gz";
        String testingFilePathCSV = "dataset/mnist_test.csv";
        
        // Alternative: Binary files in folders
        // String trainingFilePath = "dataset/mnist/train";
        // String testingFilePath = "dataset/mnist/test";
        
        ns = new TrainingState();
        network.setMode(ns);
        network.processData(trainingFilePathCSV);
        network.processData(trainingFilePathBin);
        
        ns = new TestingState();
        network.setMode(ns);
        network.processData(testingFilePathCSV);
        network.processData(testingFilePathBin);
        
        System.out.println("Experiment Complete.");
    }
}
