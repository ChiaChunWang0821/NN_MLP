import java.util.ArrayList;

public class Layer {

    protected Neuron[] neurons;
    protected Boolean isOutputLayer;
    protected ArrayList<Double> E = new ArrayList<>();

    private int currentLayerOfNeuron;
    private int previousLayerOfNeuron;

    protected Layer(int cur, int pre, Boolean flag){
        currentLayerOfNeuron = cur;
        previousLayerOfNeuron = pre;
        isOutputLayer = flag;

        neurons = new Neuron[currentLayerOfNeuron];

        for(int i = 0; i < currentLayerOfNeuron; i++){
            neurons[i] = new Neuron(previousLayerOfNeuron);
        }
    }

    protected double instantaneousErrorSquareFunction(){ // pattern learning
        if(isOutputLayer){
            double sum = 0;
            for(int i = 0; i < neurons.length; i++){
                double e = neurons[i].errorFunction();
                sum += e * e;
            }

            E.add(0.5 * sum);
            return 0.5 * sum;
        }
        return -1;
    }
}
