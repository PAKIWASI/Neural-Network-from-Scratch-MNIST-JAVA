package neuralNetwork;

public class NetworkFacade {
    private NeuralNetwork network;
    // Armaghan's State pattern will go here 

    public NetworkFacade() {

        int[] hiddenLayers = {128, 64};
        double learningRate = 0.01;
        this.network = new NeuralNetwork(hiddenLayers, learningRate);
    }

    public void runFullExperiment() {
        
        System.out.println("Starting Setup...");
        
        // Armaghan's state logic will just be a method call here:
        // stateManager.setTrainingState(); 
        network.train();
        
        // stateManager.setTestingState();
        network.test();
        
        System.out.println("Experiment Complete.");
    }
}