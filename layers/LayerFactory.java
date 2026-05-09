package layers;
import layers.strategies.*;

/**
 * Factory class to encapsulate Layer creation.
 * This ensures the correct ActivationStrategy is always paired 
 * with the correct initialization logic.
 */

public class LayerFactory {

    //Creates a standard Hidden Layer using ReLU activation.

    public static Layer createHiddenLayer(double[] input, int outputSize)
    {
        // MOVED FROM: NeuralNetwork.initLayers() (The 'new HiddenLayer' call)
        // MOVED FROM: HiddenLayer constructor (The ReLU activation assignment)
        return new Layer(input, outputSize, new RELUStrategy());
    }

    //Creates an Output Layer using Softmax activation.

    public static Layer createOutputLayer(double[] input, int outputSize)
    {
        // MOVED FROM: NeuralNetwork.initLayers() (The 'new OutputLayer' call)
        // MOVED FROM: OutputLayer constructor (The Softmax activation assignment)
        return new Layer(input, outputSize, new SoftmaxStrategy());
    }
    
}
