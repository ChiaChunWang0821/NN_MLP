import java.util.ArrayList;
import java.util.Collections;

public class MultilayerPerceptron {

    private final int EACH_LAYER_OF_NEURONS = 1;
    private final int MAX_LEARNING_CYCLE = 2;
    private final int LEARNING_RATE = 3;
    private final int MOMENTUM = 4;
    private final int THRESHOLD = 5;
    private final int ERROR_TOLERANCE = 6;

    private final static int DIMENSION = -1;
    private final static int TRAINING_SET = -2;
    private final static int TESTING_SET = -3;
    private final static int WEIGHT = -5;
    private final static int TRAINING_RECOGNITION_RATE = -6;
    private final static int TESTING_RECOGNITION_RATE = -7;

    private final static int REGRESSION = 11;
    private final static int CLASSIFICATION = 12;

    private final static double EPSILON = 0.0001;

    private Layer[] layers;

    private String[] eachLayerOfNeuronsString;
    private int[] eachLayerOfNeurons;
    private double learningRate;
    private double momentum;
    private double threshold;
    private double errorTolerance;
    private int maxLearningCycle;
    private int normalizedD;

    private double trainingRecognitionRate;
    private double testingRecognitionRate;
    private double EAV;

    private int cycle = 0;
    private int numOfClass;

    protected MultilayerPerceptron(){

    }

    protected void setValue(int type, String string){
        switch (type){
            case EACH_LAYER_OF_NEURONS:
                eachLayerOfNeuronsString = string.split(",");
                break;
            case MAX_LEARNING_CYCLE:
                maxLearningCycle = Integer.parseInt(string);
                break;
            case LEARNING_RATE:
                learningRate = Double.parseDouble(string);
                break;
            case MOMENTUM:
                momentum = Double.parseDouble(string);
                break;
            case THRESHOLD:
                threshold = Double.parseDouble(string);
                break;
            case ERROR_TOLERANCE:
                errorTolerance = Double.parseDouble(string);
                break;
            case REGRESSION:
                normalizedD = REGRESSION;
                break;
            case CLASSIFICATION:
                normalizedD = CLASSIFICATION;
                break;
        }
    }

    protected void setDataSet(OnNeuralNetworkCallback onNeuralNetworkCallback, DataSet dataSet){
        setLayers(dataSet.dimension);
        setNumOfClass(dataSet);
        normalizedD(dataSet.trainingD);
        normalizedD(dataSet.testingD);

        if(checkMaxLearningCycle(dataSet.trainingSet)){
            onNeuralNetworkCallback.setValueCallback(MAX_LEARNING_CYCLE, Integer.toString(maxLearningCycle));
        }
    }

    protected void start(OnNeuralNetworkCallback onNeuralNetworkCallback, DataSet dataSet){
        clear();

        train(dataSet);
        test(dataSet);

        updateValue(onNeuralNetworkCallback);
    }

    private void clear(){
        cycle = 0;
        EAV = 0;
    }

    private void setLayers(int dimension){
        eachLayerOfNeurons = new int[eachLayerOfNeuronsString.length + 1];
        eachLayerOfNeurons[0] = dimension;
        for(int i = 1; i < eachLayerOfNeuronsString.length; i++){
            eachLayerOfNeurons[i] = Integer.parseInt(eachLayerOfNeuronsString[i - 1]);
        }

        layers = new Layer[eachLayerOfNeurons.length];
        for (int i = 0; i < layers.length; i++){
            int cur, pre;
            Boolean outputLayer;
            if(i == 0){ // input layer
                cur = eachLayerOfNeurons[i];
                pre = 0;
                outputLayer = false;
            }
            else if(i == layers.length - 1){ // output layer
                cur = eachLayerOfNeurons[i];
                pre = eachLayerOfNeurons[i - 1];
                outputLayer = true;
            }
            else { // hidden layer
                cur = eachLayerOfNeurons[i];
                pre = eachLayerOfNeurons[i - 1];
                outputLayer = false;
            }

            layers[i] = new Layer(cur, pre, outputLayer);

            for(Neuron neuron: layers[i].neurons){
                neuron.threshold = threshold;
            }
        }
    }

    private void setNumOfClass(DataSet dataSet){
        Boolean same = false;
        ArrayList<Double> d = new ArrayList<>();
        for(double value: dataSet.trainingD){
            for(double d0: d){
                if(value == d0){
                    same = true;
                    break;
                }
            }
            if(!same){
                d.add(value);
            }
        }

        numOfClass = d.size();
    }

    private void normalizedD(ArrayList<Double> d){
        if(normalizedD == REGRESSION){
            regression(d);
        }
        else if(normalizedD == CLASSIFICATION){
            classification(d, numOfClass);
        }
    }

    private Boolean checkMaxLearningCycle(int set){
        if(set > maxLearningCycle){
            maxLearningCycle = set * 2;
            return true;
        }
        return false;
    }

    private void train(DataSet dataSet){
        int correct = 0;

        while (true){
            putDataSet(dataSet.trainingX.get(cycle), dataSet.trainingD.get(cycle));
            if(forwardPropagation()){
                correct++;
            }
            backwardPropagation();
            adjustW();
            meanSquareErrorFunction(dataSet.trainingSet);

            if(terminationCondition()){
                break;
            }

            cycle++;
        }

        trainingRecognitionRate = (double) correct / dataSet.trainingSet * 100.0;
    }

    private void putDataSet(double[] x, double d){
        Layer layer = layers[0]; // input layer
        for(int i = 0; i < layer.neurons.length; i++){
            layer.neurons[i].y = x[i];
        }

        Layer layer1 = layers[layers.length - 1]; // output layer
        for(Neuron neuron: layer1.neurons){
            neuron.d = d;
        }
    }

    private Boolean forwardPropagation(){
        Boolean allCorrect = true;

        for(int k = 1; k < layers.length; k++){ // hidden and output layer
            Layer layer = layers[k];
            for(int j = 0; j < layer.neurons.length; j++){
                double sum = 0;
                for(int i = 0; i < layer.neurons[j].numOfPreviousLayer; i++){
                    sum += layer.neurons[j].w[i] * layer.neurons[i].y;
                }
                layer.neurons[j].v = sum;
                layer.neurons[j].y = layer.neurons[j].activationFunction();
            }

            if(layer.isOutputLayer){
                for(Neuron neuron: layer.neurons){
                    if(Math.abs(neuron.y - neuron.d) > EPSILON){
                        allCorrect = false;
                        break;
                    }
                }
            }
        }

        return allCorrect;
    }

    private void backwardPropagation(){
        for(int i = 1; i < layers.length; i++){
            Layer layer = layers[i];
            if(layer.isOutputLayer){
                for(int j = 0; j < layer.neurons.length; j++){
                    layer.neurons[j].delta = (layer.neurons[j].d - layer.neurons[j].y) * layer.neurons[j].differentialActivationFunction(layer.neurons[j].v);
                }
            }
            else {
                for(int j = 0; j < layer.neurons.length; j++){
                    double sum = 0;
                    for(int k = 0; k < layer.neurons[j].w.length; k++){
                        sum += layer.neurons[k].delta * layer.neurons[k].w[j];
                    }
                    layer.neurons[j].delta = layer.neurons[j].y * (1 - layer.neurons[j].y) * (sum);
                }
            }
        }
    }

    private void adjustW(){
        for(int k = 0; k < layers.length; k++){
            Layer layer = layers[k];
            for(int j = 0; j < layer.neurons.length; j++){
                for(int i = 0; i < layer.neurons[j].w.length; i++){
                    double old = layer.neurons[j].modify_w[i];
                    double neww = learningRate * layer.neurons[j].delta * layer.neurons[i].y;
                    layer.neurons[j].modify_w[i] = (momentum * layer.neurons[j].modify_w[i]) + (learningRate * layer.neurons[j].delta * layer.neurons[i].y);
                    layer.neurons[j].w[i] = layer.neurons[j].w[i] + layer.neurons[j].modify_w[i];

                    if((old >= 0 && neww >= 0) || (old < 0 && neww < 0)){
                        momentum *= 2;
                    }
                    else {
                        momentum *= 0.5;
                    }
                }
            }
        }
    }

    private void meanSquareErrorFunction(int set){ // batch learning
        double sum = 0;
        for(Layer layer: layers){
            if(layer.isOutputLayer){
                for(int n = 0; n < layer.E.size(); n++){ // layer.E.size() or set ???
                    sum += layer.E.get(n); // when call layer's instantaneousErrorSquareFunction() ???
                }
            }
        }
        EAV = sum / set;
    }

    private Boolean terminationCondition(){
//        if(gradientVector < threshold || (gradientVector * EPSILON == 0)){
//            return true;
//        }

        if(EAV < errorTolerance){
            return true;
        }

        // 推廣能力達到目標 ???

        if(cycle > maxLearningCycle){
            return true;
        }

        return false;
    }

    private void updateValue(OnNeuralNetworkCallback onNeuralNetworkCallback){
        StringBuilder weightString = new StringBuilder(" ");
        for (Layer layer: layers){
            for(Neuron neuron: layer.neurons){
                for(int i = 1; i < neuron.w.length; i++){
                    weightString.append(neuron.w[i]).append(",");
                }
            }
        }
        weightString.deleteCharAt(weightString.length() - 1);
        onNeuralNetworkCallback.setValueCallback(WEIGHT, weightString.toString());

        onNeuralNetworkCallback.setValueCallback(TRAINING_RECOGNITION_RATE, Double.toString(trainingRecognitionRate));
        onNeuralNetworkCallback.setValueCallback(TESTING_RECOGNITION_RATE, Double.toString(testingRecognitionRate));
    }

    private void regression(ArrayList<Double> d){
        double max = Collections.max(d);
        double min = Collections.min(d);

        for(int i = 0; i < d.size(); i++){
            d.set(i, (d.get(i) - min) / (max - min));
        }
    }

    private void classification(ArrayList<Double> d, int k){
        for(Layer layer: layers){
            if(layer.isOutputLayer){
                if(layer.neurons.length == 1){
                    // 0, 1/ (k - 1), 2 / (k - 1), ..., 1 ???
                }
                else{
                    // k bits or log2(k) ???
                }
            }
        }
    }

    private void test(DataSet dataSet){
        int correct = 0;
        int count = 0;
        while(count < dataSet.testingSet){
            putDataSet(dataSet.testingX.get(count), dataSet.testingD.get(cycle));
            if(forwardPropagation()){
                correct++;
            }
            count++;
        }

        testingRecognitionRate = (double) correct / dataSet.testingSet * 100.0;
    }
}
