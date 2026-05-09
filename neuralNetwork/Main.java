package neuralNetwork;

public class Main
{

    public static void main( String[] args )
    {
        /*Old version, had manual setting up of the neural network
        double LEARNING_RATE = 0.01;

        int[] hidden = { 128, 64 };

        NeuralNetwork nn = new NeuralNetwork( hidden, LEARNING_RATE );
        
        nn.train();
        
        nn.test();*/

        //NEW version
        NetworkFacade NF=new NetworkFacade();
        NF.runFullExperiment();
    }

}
