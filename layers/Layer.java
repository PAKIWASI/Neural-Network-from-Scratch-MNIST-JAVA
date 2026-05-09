package layers; 

import java.util.Arrays;
import java.util.Random;

import layers.strategies.ActivationStrategy;
import layers.strategies.SoftmaxStrategy;

/**
 * REFACTORED: Now a concrete class using Strategy Pattern.
 * Removed: 'abstract' keyword and subclass-specific logic.
 * coverted all the protected variables to private as we dont 
 * to make subclasses now
 */
public class Layer 
{
    private double[] input;           // x      1 x m input vector from prev layer
    private double[][] weights;       // W      m x n weight matrix
    private double[] bias;            // b      1 x n bias vector
    private double[] preActOutput;  // z = xW + b  -> 1 x n
    private double[] actOutput;     // a = act(z)  -> 1 x n

    private static Random rand;

    private final int inputSize;
    private final int outputSize;

    // STRATEGY: Decouples activation math from this class
    private ActivationStrategy strategy;

    // Gradients
    private double[] dL_dz;  // local gradient dL/dz = dL/da x (da/dz)T (1 x n) x (n x 1) = n x n (here->1 x n, we work implicitly, element wise xply)  
    private double[] dL_dx;  // downstream gradient (will send to previous layer) dL/dx = dL/dz x dz/dx = dL/dz x WT  (1 x n) x (m x n)T = (1 x m)

    // Gradients to Update Weights and Biases
                                // m x n =  m x 1     1 x n
    private double[][] dL_dW;   // dL/dW = (dz/dW)T x dL/dz = (x)T x dL/dz 
    private double[] dL_db;     // dL/db = dL/dz x dz/db = dL/dz x 1     (1 x n) = (1 x n) x 1 , dz/db = 1

    // REFACTORED: Constructor now accepts the Strategy
    public Layer( double[] input, int outputSize, ActivationStrategy strategy )
    {
        rand = new Random();
        this.strategy = strategy; // MOVED FROM: Subclasses

        preActOutput = new double[ outputSize ]; // 1 x n
        actOutput = new double[ outputSize ];

        this.inputSize = input.length;
        this.outputSize = outputSize;

        this.input = input; // memory for input allocated in respective layer

        dL_dz = new double[ outputSize ];
        dL_dx = new double[ inputSize ];

        dL_dW = new double[ inputSize ][ outputSize ];
        dL_db = new double[ outputSize ];

        initWeights();
        initBiases();
    }

    public void validateInternals() {
        // Check 1: Did weights initialize?
        if (weights == null || weights[0][0] == 0 && weights[0][1] == 0) {
            throw new IllegalStateException("Layer Error: Weights are uninitialized or all zero.");
        }

        // Check 2: Is the Strategy attached?
        if (strategy == null) {
            throw new IllegalStateException("Layer Error: No ActivationStrategy assigned.");
        }

        // Check 3: Is data flowing? (Check for NaN or Dead Neurons)
        for (double val : actOutput) {
            if (Double.isNaN(val)) {
                throw new ArithmeticException("Layer Error: NaN detected in output of " + strategy.getClass().getSimpleName());
            }
        }
    }

    public void calculateOutput() // z = xW + b -> 1 x n
    {
        
        MatrixOperations.VecMatXply( input, weights, preActOutput ); // (1 x m) x (m x n) = (1 x n)


        for ( int i = 0; i < outputSize; i++ )
        
            preActOutput[ i ] += bias[ i ];

         
        // FORWARDED TO: Strategy
        strategy.forward( preActOutput, actOutput );  
        // apply activation a = act(z)  -> 1 x n
    }

    /**
     * REFACTORED: Generic gradient calculation using Strategy
     * upstreamGradient dL/da
    */
    //made public to use in neural networks class
    public void calculateLocalGradient(double[] upstreamGradient)
    {
        double[] da_dz = new double[ outputSize ];
        
        
        // 1. Get derivative from strategy (Replaces ReLuDerivitive)
        strategy.getDerivative( preActOutput, da_dz );
        // Weight Gradient
        // (m x n) = (m x 1) x (1 x n)

        // 2. Compute local gradient dL/dz = upstream * local_derivative
        MatrixOperations.vecXplyElementWise( upstreamGradient, da_dz, dL_dz );

        // 3. Compute Weight, Bias, and Downstream gradients (Original Logic)
        MatrixOperations.vecTransposeXplyVec( input, dL_dz, dL_dW ); 
        // dL/dW = (dz/dW)T x dL/dz = (x)T x dL/dz

        // Bias Gradient
        for ( int i = 0; i < outputSize; i++ )
        
            dL_db[ i ] = dL_dz[ i ];            // dL/db = dL_dz -> 1 x n
        

        // Downstream Gradient
        MatrixOperations.vecXplyMatrixTranspose( dL_dz, weights, dL_dx ); // dL/dx = dL/dz x dz/dx , dz/dx = WT -> 1 x m
    }

    public void updateWeights( double LEARNING_RATE )
    {
        for ( int i = 0; i < inputSize; i++ )   // Gradient Decent
        
            for ( int j = 0; j < outputSize; j++ ) 
            
                weights[ i ][ j ] -= LEARNING_RATE * dL_dW[ i ][ j ];  // Wij : Wij - learning rate x dL/dWij 
            
    }

    public void updateBiases( double LEARNING_RATE )
    {
        for ( int i = 0; i < outputSize; i++ )   // Gradient Decent
        
            bias[ i ] -= LEARNING_RATE * dL_db[ i ];  // bi : bi - learning rate x dL/dbi

    }


    private void initWeights()
    {
        weights = new double[ inputSize ][ outputSize ]; // m x n

        double stddev = 0.01;

        // SMELL FIXED: Removed 'if (this.getClass() == OutputLayer.class)'
        // Now using a simple check based on the strategy type or passing stddev
        if ( strategy instanceof SoftmaxStrategy )
            stddev = Math.sqrt( 2.0 / ( inputSize + outputSize ) );   // Xaviot grout init for output layer
        else
            stddev = Math.sqrt( 2.0 / inputSize );  // He initialization for ReLU

        for ( int i = 0; i < inputSize; i++ )

            for ( int j = 0; j < outputSize; j++ )

                weights[ i ][ j ] = rand.nextGaussian() * stddev;
        
        System.out.println("Weights Init" );
    }

    private void initBiases()
    {
        bias = new double[ outputSize ];  // 1 x m

        Arrays.fill( bias, 0.0 );  // zero init for biases

        System.out.println( "Biases Init" );
    }


    
    public void resetGradients()   // resetting gradients back to zero 
    {
        Arrays.fill( dL_dz, 0.0 );
        Arrays.fill( dL_dx, 0.0 );


        for ( int i = 0; i < inputSize; i++ )
            
            Arrays.fill( dL_dW[ i ], 0.0 );
        

        Arrays.fill( dL_db, 0.0 );
    }


    public void setInput( double[] input ) 
    {
        this.input = Arrays.copyOf( input, inputSize );
    }

    public double[] getOutput() { return actOutput; }


    
    public double[] getUpstreamGradient() { return dL_dx; }

    public double[] getPreActOutput() {return preActOutput;}

}
