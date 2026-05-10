package neuralNetwork;

import java.util.List;

import data.ReaderInterface;
import data.Image;

import layers.Layer;
import layers.LayerFactory;
import neuralNetwork.states.NetworkState;
import neuralNetwork.states.TestingState;

public class NeuralNetwork
{
    private Layer[] layers;                // array of all layers
    private NetworkState currentState;      // THE STATE (Mode)
    private ReaderInterface reader;         // Interface for data reading

    private double[] currInput;            // the current input the network
    private double[] currOutput;           // the current ouput from the network

    private final int inputSize = 784;     // size of mnist image vector
    private final int[] hiddenLayers;      // size of each hidden
    private final int size;                // no of hidden layers + output layer
    private final int OutputSize = 10;     // predicting 0-9 digits

    private final double LEARNING_RATE;    // constant step-size used in gradient decent

    // ReaderInterface injection
    public NeuralNetwork(int[] hidden, double LEARNING_RATE, ReaderInterface reader)
    {
        this.hiddenLayers = hidden;
        size = hiddenLayers.length + 1;       // hidden layers + 1 output layer

        this.LEARNING_RATE = LEARNING_RATE;
        this.reader = reader;                 // ADDED: Store reader reference
        
        currInput = new double[inputSize];

        initLayers();                       // assign layers by user preference
    }

    /**
     * UNIFIED PROCESSOR
     * Uses injected reader for flexible file format support
     */
    public void processData(String path) {
        if (currentState == null) {
            throw new IllegalStateException("State Error: No Mode (Training/Testing) set!");
        }
        
        // MODIFIED: Use injected reader instead of direct DataReader call
        List<Image> images = this.reader.readData(path);
        
        for (int i = 0; i < images.size(); i++) {
            // DELEGATION: The state decides how to handle the sample
            currentState.handleSample(this, images.get(i));

            // INTERNAL VALIDATION: Applied every 1000 samples regardless of state
            if (i % 1000 == 0) {
                validateLayers();
                System.out.println(currentState.getStateName() + " Progress: " + i + "/" + images.size());
            }
        }
        
        // Notify state that the data set is finished (e.g., to print accuracy)
        currentState.onStateExit();
        if(currentState instanceof TestingState)
            System.out.printf("Learning Rate: %.2f\n",LEARNING_RATE);
    }

    //MOVED FROM: private access to public
    // The state needs to trigger the forward flow.
    public void forwardPass(double[] currInput)  // z = Wx + b , a = activation(z)
    {
        layers[0].setInput(currInput);  // Set input for first layer
        
        for (int i = 0; i < layers.length; i++) 
        {
            layers[i].calculateOutput();    // calculate the output for each layer
            
            if (i < layers.length - 1) 
            
                layers[i + 1].setInput(layers[i].getOutput()); // Pass this layer's output to next layer's input
            
        }
        
        // Update current output reference
        currOutput = layers[layers.length - 1].getOutput();
    }

    public void backpropogation(double[] trueOutput)
    {
        for (Layer l : layers) l.resetGradients();  // reset the gradients computed in last iteration
        
        Layer outputLayer = layers[size - 1]; // start with output layer

        // Calculate (a - y) as the upstream gradient for Softmax + Cross-Entropy
        double[] outputError = new double[OutputSize];
        for (int i = 0; i < OutputSize; i++) {
            outputError[i] = outputLayer.getOutput()[i] - trueOutput[i];
        } 
        
        // NO CASTING NEEDED: Generic method call
        outputLayer.calculateLocalGradient(outputError);

        // 2. Backpropagate through hidden layers
        double[] upstreamGradient = outputLayer.getUpstreamGradient();  // the gradient that each layer sends to prev layer

        for (int i = size - 2; i >= 0; i--)  // backprop through all layers till input layer
        {
            // NO CASTING NEEDED: All layers are just 'Layer'
            layers[i].calculateLocalGradient(upstreamGradient);

            upstreamGradient = layers[i].getUpstreamGradient();    // set the upstream gradient for each layer 
        }
    }

    public void updateWeightsAndBiases()  // update parameters (weights and biases for each layer)
    {
        for (Layer l : layers)  // start with output layer and go till first layer
        {
            l.updateWeights(LEARNING_RATE); // update weights and biases of each
            l.updateBiases(LEARNING_RATE);
        }
    }

    private void validateLayers() {
        for (Layer l : layers) {
            l.validateInternals();
        }
    }

    private void initLayers()    // init all hidden layers + output layer and give random values to weights
    {                            //  and 0 values to biases 
        
        layers = new Layer[size];

        // 1. Create First Hidden Layer
        layers[0] = LayerFactory.createHiddenLayer(currInput, hiddenLayers[0]); // main input to network goes to first hidden

         // 2. Create number of Hidden Layers
        for (int i = 1; i < size - 1; i++)
        
            layers[i] = LayerFactory.createHiddenLayer(layers[i - 1].getOutput(), hiddenLayers[i]); // each next layer gets reference to output of prev layer as input
        
        // 3. Create Output Layer with SoftMax
        layers[size - 1] = LayerFactory.createOutputLayer(layers[size - 2].getOutput(), OutputSize);

        currOutput = layers[size - 1].getOutput();      // main output of network 
        
        for (Layer l : layers) {
            l.validateInternals(); 
        }
        System.out.println("Structure Validation: Passed.");
    }
    
    // Setters and Getters for State Interaction
    public void setMode(NetworkState state) { this.currentState = state; }
    public double[] getCurrOutput() { return currOutput; }
    public int getOutputSize() { return OutputSize; }
}
