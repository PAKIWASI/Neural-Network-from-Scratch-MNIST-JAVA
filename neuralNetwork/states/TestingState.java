package neuralNetwork.states;

import data.Image;
import neuralNetwork.NeuralNetwork;

public class TestingState implements NetworkState {

    private int correct = 0;
    private int total = 0;

    @Override
    public void handleSample(NeuralNetwork nn, Image img) {
        // MOVED FROM: NeuralNetwork.test() loop body
        nn.forwardPass(img.getData());

        double[] currOutput = nn.getCurrOutput();
        int outputSize = currOutput.length;
        
        // MOVED FROM: Max-index prediction logic in test()
        int predicted = 0;
        double maxProb = currOutput[ 0 ];
        for ( int j = 1; j < outputSize; j++ ) 
            {
                if ( currOutput[j] > maxProb ) 
                    {
                        maxProb = currOutput[ j ];
                        predicted = j;           // the j with max prob is the prediction
                    }
            }
        
        if (img.getLabel() == predicted) 
            correct++;
        total++;
    }

    @Override
    public void onStateExit() 
    {
        // MOVED FROM: Final accuracy print in NeuralNetwork.test()
        
        double acc = (correct * 100.0) / total;
        System.out.println("\n====================================");
        System.out.printf("Correct Predictions: %d\nTest Accuracy: %.2f %%\n",correct,acc);
    }

    //getters for the total and correct testing samples

    public int getTotal() {return total;}

    public int getCorrect() {return correct;}

    @Override
    public String getStateName(){ return "TEST";}
}
    

