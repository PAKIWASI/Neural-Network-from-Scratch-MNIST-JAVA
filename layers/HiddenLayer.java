package layers;


public class HiddenLayer extends Layer
{
    // Gradients
    private double[] dL_da;  // upstream gradient (will receive from next layer) 1 x n vector



    public HiddenLayer( double[] input, int outputSize )
    {
        super( input, outputSize );
    }


                               // dL/da is upstream gradient
    public void calculateLocalGradient( double[] upstreamGradient )
    {
        this.dL_da = upstreamGradient;  // 1 x n

        double[] da_dz = new double[ outputSize ]; // intermediate derivitive, needed to calc dL/dz
        ReLuDerivitive( preActOutput, da_dz );   // da_dz = d/dz (a(z)) = a'(z)  -> 1 x n


        // Local Gradient
        MatrixOperations.vecXplyElementWise( dL_da, da_dz, dL_dz );  // dL/dz = dL/da x da/dz (element-wise xply) -> 1 x n 
                                                                            

        super.calculateLocalGradient();  // get dL/dW, dL/b, dL/dx
        
    }

    @Override     // ReLu Activation  a = relu(z)   -> 1 x n
    protected void activation( double[] preActOutput, double[] actOutput )
    {
        for ( int i = 0; i < outputSize; i++ )
        {
            if ( preActOutput[ i ] <= 0 )

                actOutput[ i ] = 0;

            else

                actOutput[ i ] = preActOutput[ i ];
        }
    }


    private void ReLuDerivitive( double[] preActOutput, double[] da_dz ) 
    {
        
        for ( int i = 0; i < outputSize; i++ )
        {
            if ( preActOutput[ i ] <= 0 )

                da_dz[ i ] = 0;

            else

                da_dz[ i ] = 1;
        }
    }


}
