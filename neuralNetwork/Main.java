package neuralNetwork;



public class Main
{

    public static void main( String[] args )
    {
        double LEARNING_RATE = 0.01;

        int[] hidden = { 128, 64 };

        NeuralNetwork newNet = new NeuralNetwork( hidden, LEARNING_RATE );
        
        newNet.train();
        
        newNet.test();
    }

    

}
