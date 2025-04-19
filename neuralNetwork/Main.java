package neuralNetwork;



public class Main
{

    public static void main(String[] args)
    {
        double LEARNING_RATE = 0.01;

        int[] hidden = { 128, 64 };

        NeuralNetwork nn = new NeuralNetwork(hidden, LEARNING_RATE);
        
        nn.train();
        nn.test();
    }
    

}
