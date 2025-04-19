package layers;

import java.util.Arrays;
import java.util.Random;


public abstract class Layer 
{
    private double[] input;           // x      1 x m input vector from prev layer
    private double[][] weights;       // W      m x n weight matrix
    private double[] bias;            // b      1 x n bias vector
    protected double[] preActOutput;  // z = xW + b  -> 1 x n
    protected double[] actOutput;     // a = act(z)  -> 1 x n

    private static Random rand;

    protected final int inputSize;
    protected final int outputSize;


    // Gradients
    protected double[] dL_dz;  // local gradient dL/dz = dL/da x (da/dz)T (1 x n) x (n x 1) = n x n (here->1 x n, we work implicitly, element wise xply)  
    protected double[] dL_dx;  // downstream gradient (will send to previous layer) dL/dx = dL/dz x dz/dx = dL/dz x WT  (1 x n) x (m x n)T = (1 x m)

    // Gradients to Update Weights and Biases
                                // m x n =  m x 1     1 x n
    protected double[][] dL_dW;   // dL/dW = (dz/dW)T x dL/dz = (x)T x dL/dz 
    protected double[] dL_db;     // dL/db = dL/dz x dz/db = dL/dz x 1     (1 x n) = (1 x n) x 1 , dz/db = 1


    public Layer( double[] input, int outputSize )
    {
        rand = new Random();

        preActOutput = new double[ outputSize ]; // 1 x n
        actOutput = new double[ outputSize ];

        this.inputSize = input.length;
        this.outputSize = outputSize;

        this.input = input; // memory for input allocated in respective layer

        dL_dz = new double[outputSize];
        dL_dx = new double[inputSize];

        dL_dW = new double[inputSize][outputSize];
        dL_db = new double[outputSize];

        initWeights();
        initBiases();
    }

    public void calculateOutput() // z = xW + b -> 1 x n
    {
        
        MatrixOperations.VecMatXply( input, weights, preActOutput ); // (1 x m) x (m x n) = (1 x n)


        for ( int i = 0; i < outputSize; i++ )
        
            preActOutput[ i ] += bias[ i ];

         
        activation( preActOutput, actOutput );  // apply activation a = act(z)  -> 1 x n
    }


    protected abstract void activation( double[] preActOutput, double[] actOutput ); //activation function (unque to layer type)
    

    protected void calculateLocalGradient()
    {
        // Weight Gradient
                                        // (m x n) = (m x 1) x (1 x n)
        MatrixOperations.vecTransposeXplyVec(input, dL_dz, dL_dW); // dL/dW = (dz/dW)T x dL/dz = (x)T x dL/dz

        // Bias Gradient
        for ( int i = 0; i < outputSize; i++)
        
            dL_db[i] = dL_dz[i];            // dL/db = dL_dz -> 1 x n
        

        // Downstream Gradient
        MatrixOperations.vecXplyMatrixTranspose(dL_dz, weights, dL_dx); // dL/dx = dL/dz x dz/dx , dz/dx = WT -> 1 x m
        
    }

    public void updateWeights( double LEARNING_RATE )
    {
        for (int i = 0; i < inputSize; i++)   // Gradient Decent
        
            for (int j = 0; j < outputSize; j++)
            
                weights[i][j] -= LEARNING_RATE * dL_dW[i][j];  // Wij : Wij - learning rate x dL/dWij 
            
    }

    public void updateBiases( double LEARNING_RATE)
    {
        for (int i = 0; i < outputSize; i++)   // Gradient Decent
        
            bias[i] -= LEARNING_RATE * dL_db[i];  // bi : bi - learning rate x dL/dbi

    }


    protected void initWeights()
    {
        weights = new double[inputSize][outputSize]; // m x n

        double stddev = 0.01;

        if (this.getClass() == OutputLayer.class)
            stddev = Math.sqrt(2.0 / (inputSize + outputSize));   // Xaviot grout init for output layer
        else
            stddev = Math.sqrt(2.0 / inputSize);  // He initialization for ReLU

        for (int i = 0; i < inputSize; i++)

            for (int j = 0; j < outputSize; j++)

                weights[i][j] = rand.nextGaussian() * stddev;
        
        System.out.println("x");
    }

    private void initBiases()
    {
        bias = new double[outputSize];  // 1 x m

        Arrays.fill(bias, 0.0);  // zero init for biases
    }


    
    public void resetGradients()   // resetting gradients back to zero 
    {
        Arrays.fill(dL_dz, 0.0);
        Arrays.fill(dL_dx, 0.0);


        for (int i = 0; i < inputSize; i++)
            
            Arrays.fill(dL_dW[i], 0.0);
        

        Arrays.fill(dL_db, 0.0);
    }


    public void setInput(double[] input) 
    {
        this.input = Arrays.copyOf(input, input.length);
    }

    public double[] getOutput() { return actOutput; }


    
    public double[] getUpstreamGradient() { return dL_dx; }

}
