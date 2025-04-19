package neuralNetwork;

import java.util.List;

import data.DataReader;
import data.Image;
import layers.HiddenLayer;
import layers.Layer;
import layers.OutputLayer;


public class NeuralNetwork
{
    
    private Layer[] layers;                // array of all layers

    private List< Image > images;          // training / testing images
    private double[] currInput;            // the current input to the network
    private double[] currOutput;           // the current ouput from the network

    private final int inputSize = 784;     // size of mnist image vector
    private final int[] hiddenLayers;      // size of each hidden
    private final int size;                // no of hidden layers + output layer
    private final int OutputSize = 10;     // predicting 0-9 digits

    private final double LEARNING_RATE;    // constant step-size used in gradient decent (by how much we decend each iteration)




    public NeuralNetwork( int[] hidden, double LEARNING_RATE )
    {
        this.hiddenLayers = hidden;
        size = hiddenLayers.length + 1;       // hidden layers + 1 output layer

        this.LEARNING_RATE = LEARNING_RATE;
        
        currInput = new double[ inputSize ];

        initLayers();                       // assign layers by user preference
    }


    public void train()     // train model on mnist data  
    {                     // get images from dataReader 
        images = DataReader.readData("D:\\Programming Projects\\Java Programmes\\Neural-Network\\src\\data\\mnist_train.csv");
    
        for ( int i = 0; i < images.size(); i++ ) 
        {
            System.out.println( "Training sample: " + i );

            int label = images.get( i ).getLabel();         // True label (0-9)
            currInput = images.get( i ).getData();          // Input to the network
    
            double[] trueOutput = new double[ OutputSize ]; // hot coded vector (arr[trueLabel] =  1 else 0)
            trueOutput[ label ] = 1;
            
            forwardPass();               // get output for each input

            backpropogation( trueOutput ); // get error(Loss->scalar), compute gradients (calculate by how much the network is wrong)
            
            updateWeightsAndBiases();    // update the parameters based on the the Loss by gradient decent algorithm 
        }
        
    }

    public void test()   // test the network on seperate data
    {
        images = DataReader.readData("D:\\Programming Projects\\Java Programmes\\Neural-Network\\src\\data\\mnist_test.csv");
        
        int total = images.size();
        int correct = 0;
        
        for ( int i = 0; i < images.size(); i++ ) 
        {
            System.out.println( "Testing sample: " + i );

            int label = images.get( i ).getLabel();   // True label (0-9)
            currInput = images.get( i ).getData();    // Input to the network
    
            
            forwardPass();                 // get output for each input
            
            // Find predicted label (index of max output probability)
            int predicted = 0;
            double maxProb = currOutput[ 0 ];
            for ( int j = 1; j < OutputSize; j++ ) 
            {
                if ( currOutput[j] > maxProb ) 
                {
                    maxProb = currOutput[ j ];
                    predicted = j;           // the j with max prob is the prediction
                }
            }
    
            // Check if prediction matches true label
            if ( label == predicted )
                correct++;
        }
    
        double accuracy = ( correct * 100.0 ) / total;
        
        // Calculate and print accuracy
        System.out.println();
        System.out.println("====================================");
        
        System.out.println( "Total test samples: " + total );
        System.out.println( "Correct predictions: " + correct );
        System.out.println( "Accuracy: " + accuracy + "%" );
        System.out.println( "Learning Rate: " + LEARNING_RATE );

        System.out.println("====================================");
        System.out.println();
    }

    private void forwardPass()  // z = Wx + b , a = activation(z)
    {
        
        layers[ 0 ].setInput( currInput );  // Set input for first layer
        
        for ( int i = 0; i < layers.length; i++ ) 
        {
            layers[ i ].calculateOutput();    // calculate the output for each layer
            
            if ( i < layers.length - 1 ) 
            
                layers[ i + 1 ].setInput( layers[ i ].getOutput() ); // Pass this layer's output to next layer's input
            
        }
    }

    private void backpropogation( double[] trueOutput )
    {
        for ( Layer l : layers ) l.resetGradients();  // reset the gradients computed in last iteration

        
        OutputLayer o = ( OutputLayer ) layers[ size - 1 ]; // start with output layer
        o.calculateLocalGradient( trueOutput );           // compute the local gradient and all other gradients needed to update parameters

        double[] upstreamGradient = o.getUpstreamGradient();  // the gradient that each layer sends to prev layer


        for ( int i = size - 2; i >= 0; i-- )  // backprop through all layers till input layer
        {
            HiddenLayer h = ( HiddenLayer ) layers[ i ];
            h.calculateLocalGradient( upstreamGradient );

            upstreamGradient = h.getUpstreamGradient();    // set the upstream gradient for each layer 
        }
    }

    private void updateWeightsAndBiases()  // update parameters (weights and biases for each layer)
    {
        for ( Layer l : layers )  // start with output layer and go till first layer
        {
            l.updateWeights( LEARNING_RATE ); // update weights and biases of each
            l.updateBiases( LEARNING_RATE );
        }
    }

    private void initLayers()    // init all hidden layers + output layer and give random values to weights
    {                            //  and 0 values to biases 
        
        layers = new Layer[ size ];

        
        layers[ 0 ] = new HiddenLayer( currInput, hiddenLayers[ 0 ] ); // main input to network goes to first hidden


        for ( int i = 1; i < size - 1; i++ )
        
            layers[ i ]  = new HiddenLayer( layers[ i - 1 ].getOutput(), hiddenLayers[ i ] ); // each next layer gets reference to output of prev layer as input
        

        layers[ size - 1 ] = new OutputLayer( layers[ size - 2 ].getOutput(), OutputSize );

        currOutput = layers[ size - 1 ].getOutput();      // main output of network 
    }
}