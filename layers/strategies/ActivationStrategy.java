package layers.strategies;

/**
 * Strategy Pattern: Decouples activation math from Layer logic.
 * Previously, this logic was hardcoded in HiddenLayer and OutputLayer subclasses.
 */
public interface ActivationStrategy {
    // Forward Pass: a = f(z)
    void forward(double[] z, double[] a);

    // Provides the derivative: da/dz = f'(z)
    void getDerivative(double[] z, double[] da_dz);
}
//the two functions cannot be seperate to differnt 
// interfaces as they aretwo halves of a single 
// mathematical responsibility