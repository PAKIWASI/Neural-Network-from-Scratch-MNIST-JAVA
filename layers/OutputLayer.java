package layers;


public class OutputLayer extends Layer
{


    public OutputLayer( double[] input, int outputSize )
    {

        super( input, outputSize );

    }


    public void calculateLocalGradient( double[] trueOutput ) // Loss L is a scalar values, trueOutput (1 x n) hot coded vector
    {
        // Local Gradient

        for ( int i = 0; i < outputSize; i++ )
        
            dL_dz[ i ] = actOutput[ i ] - trueOutput[ i ]; // derivative of softmax + cross entropy loss (1 x n)
                                                     // dont need dL/da for output layer, compute dL/dz directly

        super.calculateLocalGradient();
    }


    @Override                                               // SOFTMAX ACTIVATION a = softmax(z) -> 1 x n
    protected void activation( double[] preActOutput, double[] actOutput ) 
    {
        double max = Double.NEGATIVE_INFINITY;
        
                                                    // Find maximum value for numerical stability
        for ( int i = 0; i < outputSize; i++ )
            
            if ( preActOutput[ i ] > max )
                
                max = preActOutput[ i ];
            
        
        double sum = 0.0;
        
                                            // Compute exponentials (shifted by max to avoid overflow)
        for ( int i = 0; i < outputSize; i++ ) 
        {
            actOutput[ i ] = Math.exp( preActOutput[ i ] - max );
            sum += actOutput[ i ];
        }
        
                                             // Normalize to probabilities
        for ( int i = 0; i < outputSize; i++ )
            
            actOutput[ i ] /= sum;
    }

    

    
}
