package layers.strategies;

public class RELUStrategy implements ActivationStrategy {

    @Override
    // MOVED FROM: HiddenLayer.activation()
    public void forward(double[] z, double[] a)
    { // ReLu Activation  a = relu(z)   -> 1 x n
        int outputSize = z.length;

        for ( int i = 0; i < outputSize; i++ )
            {
                if ( z[ i ] <= 0 )

                    a[ i ] = 0;

                else

                a[ i ] = z[ i ];
            }
    }

    @Override
    // Moved from HiddenLayer.RELUDerivative()
    public void getDerivative(double[] z, double[] da_dz)
    {
        //calculates the RELU derivative is z[i] <=0 , drivative is also zero
        int outputSize = z.length;

        for ( int i = 0; i < outputSize; i++ )
        {
            if ( z[ i ] <= 0 )

                da_dz[ i ] = 0;

            else

                da_dz[ i ] = 1;
        }

    }


}
