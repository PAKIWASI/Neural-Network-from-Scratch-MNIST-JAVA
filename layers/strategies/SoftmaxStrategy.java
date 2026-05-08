package layers.strategies;

public class SoftmaxStrategy implements ActivationStrategy{

    @Override
    public void forward(double[] z, double[] a) {
        // MOVED FROM: OutputLayer.activation()
        int outputSize = z.length;
        double max = Double.NEGATIVE_INFINITY;
        
        // Find maximum value for numerical stability
        for ( int i = 0; i < outputSize; i++ )
            
            if ( z[ i ] > max )
                
                max = z[ i ];
            
        
        double sum = 0.0;
        
        // Compute exponentials (shifted by max to avoid overflow)
        for ( int i = 0; i < outputSize; i++ ) 
        {
            a[ i ] = Math.exp( z[ i ] - max );
            sum += z[ i ];
        }
        
        // Normalize to probabilities
        for ( int i = 0; i < outputSize; i++ )
            
            a[ i ] /= sum;
    }

    @Override
    public void getDerivative(double[] z, double[] da_dz) {
        /* * REFACTOR NOTE: For Softmax + Cross-Entropy, the math simplifies 
         * such that we calculate dL/dz directly as (a - y).
         * By returning 1.0 here, the generic Layer.java logic 
         * (dL/dz = upstream * da_dz) remains mathematically correct.
         */
        for (int i = 0; i < da_dz.length; i++) {
            da_dz[i] = 1.0; 
        }
    }
}