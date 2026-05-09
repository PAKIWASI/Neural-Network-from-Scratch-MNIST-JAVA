package neuralNetwork.states;
import neuralNetwork.NeuralNetwork;
import data.Image;

public interface NetworkState {

    void handleSample(NeuralNetwork nn, Image img);

    public String getStateName();

    void onStateExit();// for generating stats
}
