package layers;

import java.util.Arrays;

public class MatrixOperations
{

                                  // (1 x m) x (m x n) = (1 x n)
    public static void VecMatXply( double[] vector, double[][] matrix, double[] output )
    {
        int m = vector.length;
        int mm = matrix.length;
        int n = matrix[0].length;

        Arrays.fill(output, 0.0);

        if ( m != mm )
        {
            System.err.println("Dimentions dont add up");
            return;
        }


        for ( int j = 0; j < n; j++ )
        
            for (int i = 0; i < m; i++ )
            
                output[ j ] += vector[ i ] * matrix[ i ][ j ];
            
    }
    
                                    //         vec1                 vec2           output -> 1 x n
    public static void vecXplyElementWise( double[] dL_da, double[] da_dz, double[] dL_dz )
    {
        int n1 = dL_da.length;
        int n2 = da_dz.length;
        int n3 = dL_dz.length;

        if ( n1 != n2 && n1 != n3 && n2 != n3 )
        {
            System.err.println("Vectors have different length");
            return;
        }

        for ( int i = 0; i < dL_dz.length; i++ )

            dL_dz[ i ] = dL_da[ i ] * da_dz[ i ];
    }

                                        // dL/dW = (dz/dW)T x dL/dz = (x)T x dL/dz
    public static void vecTransposeXplyVec( double[] input, double[] dL_dz, double[][] dL_dW ) // (m x n) = (m x 1) x (1 x n)
    {
        int m = input.length;
        int n = dL_dz.length;

        for ( int i = 0; i < m; i++ )
        
            for ( int j = 0; j < n; j++ )
            
                dL_dW[ i ][ j ] = input[ i ] * dL_dz[ j ];

    }
                     
                       // dL/dx = dL/dz x dz/dx , dz/dx = WT                                      
    public static void vecXplyMatrixTranspose( double[] dL_dz, double[][] weights, double[] dL_dx ) // (1 x n) x (m x n)T = 1 x m
    {
        int n = dL_dz.length;
        int m = weights.length;
        int nn = weights[0].length;
        
        if ( n != nn )
        {
            System.err.println("Dimentions dont add up T");
            return;
        }

        Arrays.fill( dL_dx, 0.0 );

        for ( int i = 0; i < m; i++ )

            for ( int j = 0; j < n; j++ ) 
            
                dL_dx[ i ] += dL_dz[ j ] * weights[ i ][ j ];
    }
}