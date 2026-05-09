package neuralNetwork.states;

import data.Image;
import neuralNetwork.NeuralNetwork;

public class TrainingState implements NetworkState {

    @Override
    public void handleSample(NeuralNetwork nn, Image img)
    {
        final int outputSize = 10;
        // MOVED FROM: NeuralNetwork.train() loop body
        double[] trueOutput = new double[outputSize];
        trueOutput[img.getLabel()] = 1.0;

        nn.forwardPass(img.getData());
        nn.backpropogation(trueOutput);
        nn.updateWeightsAndBiases();

    }

    @Override
    public void onStateExit()
    {
        System.out.println("Training Phase Finished.");
    }

    @Override
    public String getStateName(){ return "TRAIN";}


    
}
