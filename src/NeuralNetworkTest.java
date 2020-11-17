public class NeuralNetworkTest{

    public static void main(String[] args)
    {
        testMLP();
    }

    private static void testMLP(){
        NeuralNetworkJFrame neuralNetworkJFrame = new NeuralNetworkJFrame();
        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();

        OnNeuralNetworkCallback onNeuralNetworkCallback = new OnNeuralNetworkCallback() {
            DataSet set = null;

            @Override
            public void dataReadyCallback(DataSet dataSet) {
                set = dataSet;
                multilayerPerceptron.setDataSet((OnNeuralNetworkCallback)this, dataSet);
            }

            @Override
            public void showResultCallback() {
                if (set != null){
                    multilayerPerceptron.setDataSet((OnNeuralNetworkCallback)this, set);
                    multilayerPerceptron.start((OnNeuralNetworkCallback)this, set);
                }
            }

            @Override
            public void getValueCallback(int type, String string) {
                multilayerPerceptron.setValue(type, string);
            }

            @Override
            public void setValueCallback(int type, String string) {
                neuralNetworkJFrame.updateValue(type, string);
            }
        };

        neuralNetworkJFrame.actionListener(onNeuralNetworkCallback);
    }
}
